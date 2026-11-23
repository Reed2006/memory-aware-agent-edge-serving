"""Resumable Phase 2 campaign: run at night, cut it any time, resume in the same environment.

Each session verifies the locked environment (llama-server build, model, code,
Python packages, workloads), stops any stale backend, deletes only snapshot files
this project minted, starts a fresh backend and advances through the stages,
skipping everything already finished. A run that is cut short is quarantined by
the benchmark and redone next session; it never enters a summary.

A watchdog aborts the current run when the charger is unplugged or the machine
sleeps (lid closed), because timings across either event are not comparable.
Nothing here changes system settings.
"""

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable

import psutil

from phase2.bench import power_state
from phase2.costs import CostModel
from phase2.freeze import select_shared
from phase2.plan import run_done, task_work, terminal_record
from phase2.state import PolicyConfig

PROJECT = Path(__file__).resolve().parent.parent
POINTER = PROJECT / '.phase2_campaign'
DEV_WORKLOAD = PROJECT / 'runs/final/workload_dev.json'
EVAL_WORKLOAD = PROJECT / 'runs/final/workload_eval.json'
MODEL = PROJECT / 'models/Qwen3-4B-Q4_K_M.gguf'
MODEL_SHA = PROJECT / 'runs/final/model.sha256'
PINNED_COMMIT = '620d2225c'
SOURCES = ['phase1/*.py', 'phase2/*.py', 'scripts/phase2.sh', 'scripts/phase2_night.sh',
           'requirements-lock.txt']
PRESETS = {
    # Per-platform settings fixed at init. Minutes per task are Phase 1 FIFO rates.
    # The snapshot budget must equal the one dev tested (PolicyConfig default, 4 GiB).
    'm4': dict(arrival='scaled:1', aging_grid='20,40,80', minutes_per_task=1.7, speed=1.0,
               checkpoint_gib=4.0),
    't4': dict(arrival='scaled:3.5', aging_grid='4,8,16', minutes_per_task=.48, speed=.3,
               checkpoint_gib=4.0),
}
EXPERIMENT_REVISION = 2
SEEDS = dict(waits='100', smoke='1', dev='101,102,103', eval='1101,1102,1103', sensitivity='2101,2102')
SERVERS = {'normal': dict(cache_ram=0, idle=False, directory='backend', group='P2-C24576-RAM0-idle0'),
           'ram': dict(cache_ram=3072, idle=True, directory='backend-ram', group='P2-C24576-RAM3072-idle1')}
# M4 minutes for each probe suite; scaled by the preset's speed.
PROBES = [('affinity', 5), ('incremental-recompute', 70), ('restore', 30), ('interference', 40)]
KEY_LABELS = ['fifo', 'best-fixed', 'affinity-fixed-best']  # plus the frozen final system
# Core repeats: trials 0..CONFIRM_TRIALS-1 of the key labels (one added trial on M4).
CONFIRM_TRIALS = 2
# Sensitivity tests the waits a cost lease prices: the recorded (millisecond) tool
# waits and a 15 s wait that a TTL tuned on dev waits does not cover.
SENSITIVITY = 'gap-recorded,gap-long'
# A fixed TTL tuned on dev waits is the strongest simple rival of a cost lease; only
# changed waits and loads (sensitivity) can show whether the lease adapts and it does not.
SENSITIVITY_EXTRA = ['fixed-ttl']
# Reuse from a pilot campaign (init --reuse-from). Revision 2 changed lease pricing,
# online cost updates and checkpointing. These policies never price a lease, never
# checkpoint and never read the cost model; none of the pilot's code changes after
# its first dev run touched their path (see its lock-history). Their pilot dev runs
# are therefore the runs revision 2 would make. Probes and wait samples measure the
# backend and the replay, not the scheduler.
REUSABLE_POLICIES = ('fifo', 'fixed', 'affinity', 'affinity-fixed', 'ram-cache')
REUSABLE_SETUP = {  # stage -> what it produced
    'probe-affinity': 'probes/affinity', 'probe-incremental-recompute': 'probes/incremental-recompute',
    'probe-restore': 'probes/restore', 'probe-interference': 'probes/interference',
    'probe-ram-cache': 'probes/ram-cache', 'collect-waits': 'waits',
    'calibrate-probes': 'profiles/probe.json', 'calibrate': 'profiles/profile.json',
}
# Everything a reused measurement depends on except the scheduler source.
SAME_ENVIRONMENT = ('llama_server', 'model_sha256', 'python', 'packages_sha256', 'workloads')
# The first measured run of a session would otherwise get a cool machine: on the fanless
# M4 it ran ~15-30% faster than runs after it (dev, 2026-11-17). A short unmeasured
# replay brings the machine to its working temperature first.
WARMUP_MINUTES = 6
MINTED = ('phase2-', 'probe-')  # snapshot names created by phase2.checkpoints / phase2.probes
WATCH_S = 15.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def llama_version() -> str:
    try:
        out = subprocess.run(['llama-server', '--version'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return 'unavailable'
    lines = (out.stdout + out.stderr).splitlines()
    return ' | '.join(line.strip() for line in lines if 'version' in line or 'built with' in line)


def fingerprint() -> dict:
    """Everything that must stay identical between sessions of one campaign."""
    sources = {str(path.relative_to(PROJECT)): sha256(path)
               for pattern in SOURCES for path in sorted(PROJECT.glob(pattern))}
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    return dict(llama_server=llama_version(), model_sha256=sha256(MODEL),
                python=sys.version.split()[0],
                packages_sha256=hashlib.sha256('\n'.join(packages).encode()).hexdigest(),
                workloads={str(p.relative_to(PROJECT)): sha256(p) for p in (DEV_WORKLOAD, EVAL_WORKLOAD)},
                sources=sources)


def differences(locked: dict, current: dict) -> list[str]:
    changes = []
    for key in sorted(set(locked) | set(current)):
        if key == 'sources':
            old, new = locked.get(key, {}), current.get(key, {})
            changes += [f'code changed: {name}' for name in sorted(set(old) | set(new))
                        if old.get(name) != new.get(name)]
        elif locked.get(key) != current.get(key):
            changes.append(f'{key}: {locked.get(key)!r} -> {current.get(key)!r}')
    return changes


def git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=PROJECT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(head=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')))


def validate_pinned_environment(current: dict) -> None:
    expected = MODEL_SHA.read_text().split()[0] if MODEL_SHA.exists() else None
    if expected and current['model_sha256'] != expected:
        raise SystemExit('model file does not match runs/final/model.sha256')
    if PINNED_COMMIT not in current['llama_server']:
        raise SystemExit(f'llama-server is not the pinned build ({PINNED_COMMIT}): {current["llama_server"]}')


def parse_until(text: str | None, hours: float | None, now: dt.datetime) -> float | None:
    if hours is not None:
        return (now + dt.timedelta(hours=hours)).timestamp()
    if not text:
        return None
    clock = dt.datetime.strptime(text, '%H:%M').time()
    target = dt.datetime.combine(now.date(), clock)
    if target <= now:
        target += dt.timedelta(days=1)  # "07:30" at 23:00 means tomorrow morning
    return target.timestamp()


@dataclass
class Stage:
    name: str
    kind: str                      # once | manifest | gate
    server: str | None = None      # normal | ram | None (model-free)
    minutes: float = 0.            # once: whole stage; manifest: one run (before speed scaling)
    command: Callable[[], list[str]] | None = None
    output: Path | None = None     # once: done when this file exists
    manifest: Path | None = None
    group: str | None = None
    restart: str = 'never'
    fresh: Path | None = None      # once: logs of an interrupted attempt are moved aside
    expected_runs: int = 0         # manifest: planned runs before the manifest exists


class Campaign:
    def __init__(self, root: Path, *, runner=None, fingerprinter=fingerprint, power=power_state,
                 clock=time.time, python=sys.executable):
        self.root = Path(root).resolve()
        self.state = self.root / 'campaign'
        self.runner = runner or self._spawn
        self.fingerprinter, self.power, self.clock = fingerprinter, power, clock
        self.python = python
        self.child: subprocess.Popen | None = None
        self.abort: str | None = None
        self.server_kind: str | None = None
        self.log_path = self.state / 'campaign.log'

    # ----- persistent files -------------------------------------------------
    @property
    def lock(self) -> dict:
        return json.loads((self.state / 'lock.json').read_text())

    @property
    def config(self) -> dict:
        return self.lock['config']

    def note(self, message: str) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self.log_path.open('a') as log:
            log.write(f'[{stamp}] {message}\n')
        print(f'[{stamp}] {message}', flush=True)

    def _append(self, name: str, row: dict) -> None:
        with (self.state / name).open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def init(self, preset: str, platform_id: str, reuse_from: Path | None = None) -> None:
        if (self.state / 'lock.json').exists():
            raise SystemExit(f'{self.state / "lock.json"} exists; this campaign is already initialised')
        if self.root.exists() and any(self.root.iterdir()):
            raise SystemExit('a new campaign requires an empty results directory; preserve the old pilot')
        if preset not in PRESETS:
            raise SystemExit(f'unknown platform preset {preset}; choose {sorted(PRESETS)}')
        current = self.fingerprinter()
        validate_pinned_environment(current)
        config = dict(PRESETS[preset], preset=preset, platform_id=platform_id, seeds=dict(SEEDS),
                      experiment_revision=EXPERIMENT_REVISION, sensitivity_conditions=SENSITIVITY)
        # Validate the whole import before writing anything.
        reuse = self._pilot_import(Path(reuse_from), config, current) if reuse_from else None
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / 'lock.json').write_text(json.dumps(dict(
            config=config, environment=current, created=dt.datetime.now().isoformat(),
            git=git_state()), indent=2) + '\n')
        POINTER.write_text(str(self.root) + '\n')
        self.note(f'initialised campaign for {platform_id} ({preset})')
        if reuse:
            self._copy_import(reuse)

    def _pilot_import(self, pilot: Path, config: dict, current: dict) -> dict:
        """What a pilot campaign contributes; refuses anything a changed input could affect."""
        pilot = pilot.resolve()
        try:
            lock = json.loads((pilot / 'campaign/lock.json').read_text())
            manifest = json.loads((pilot / 'manifests/dev.json').read_text())
        except (OSError, ValueError) as error:
            raise SystemExit(f'{pilot} is not a campaign with a dev manifest: {error}') from error
        old = lock['config']
        for key in ('preset', 'platform_id', 'arrival'):
            if old.get(key) != config[key]:
                raise SystemExit(f'pilot {key} {old.get(key)!r} differs from {config[key]!r}')
        for key in ('dev', 'waits'):
            if old['seeds'][key] != config['seeds'][key]:
                raise SystemExit(f'pilot {key} seeds {old["seeds"][key]} differ from {config["seeds"][key]}')
        changed = [key for key in SAME_ENVIRONMENT if lock['environment'].get(key) != current.get(key)]
        if changed:
            raise SystemExit('the pilot measured a different ' + ', '.join(changed) + '; nothing can be reused')
        missing = [path for path in REUSABLE_SETUP.values() if not (pilot / path).exists()]
        if missing:
            raise SystemExit(f'the pilot has no {", ".join(missing)}')
        profile = pilot / 'profiles/profile.json'
        if manifest['cost_profile']['sha256'] != sha256(profile):
            raise SystemExit('pilot dev runs used another cost profile than profiles/profile.json')
        # The profile is copied byte for byte so the reused runs keep their cost
        # hash. Settings it predates must default to what calibration now writes.
        costs = CostModel.from_file(profile)
        if (costs.prefill_min_tokens, costs.online_prefill, costs.online_io) != (128, True, True):
            raise SystemExit('pilot profile online-update settings differ from the current calibration')
        labels = set(self._screen_labels(config))
        runs = []
        for run in manifest['runs']:
            if run['policy'] not in REUSABLE_POLICIES or run['label'] not in labels:
                continue
            record = terminal_record(run, manifest)
            if record is None or record.get('failed', 0) or record.get('completed') != record.get('tasks'):
                continue  # unfinished or failed in the pilot: run it here
            if PolicyConfig.from_dict(run['config']).param_hash != run['param_hash']:
                raise SystemExit(f'{run["label"]}: its configuration hashes differently under this code')
            source = Path(run['results_dir']).resolve()
            try:
                target = source.relative_to(pilot)
            except ValueError as error:
                raise SystemExit(f'{run["label"]} results are outside the pilot directory') from error
            runs.append(dict(label=run['label'], policy=run['policy'], seed=run['seed'], trial=run['trial'],
                             param_hash=run['param_hash'], source=str(source), target=str(target)))
        return dict(source=str(pilot), profile_sha256=sha256(profile), runs=runs,
                    code_changes=differences(lock['environment'], current))

    def _copy_import(self, reuse: dict) -> None:
        pilot = Path(reuse['source'])
        for stage, path in REUSABLE_SETUP.items():
            source, target = pilot / path, self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            (self.state / 'done').mkdir(parents=True, exist_ok=True)
            (self.state / 'done' / stage).write_text(f'imported from {source}\n')
        for run in reuse['runs']:
            shutil.copytree(run['source'], self.root / run['target'])
        record = dict(time=dt.datetime.now().isoformat(), stages=sorted(REUSABLE_SETUP),
                      policies=list(REUSABLE_POLICIES), **reuse)
        (self.state / 'imported.json').write_text(json.dumps(record, indent=2) + '\n')
        self._append('lock-history.jsonl', dict(
            time=record['time'], reason=f'imported probes, wait samples, profiles and '
            f'{len(reuse["runs"])} dev runs of policies unchanged by this revision from {pilot}',
            changes=reuse['code_changes']))
        self.note(f'imported {len(reuse["runs"])} dev runs and {len(REUSABLE_SETUP)} setup stages from {pilot} '
                  '(see campaign/imported.json)')

    def relock(self, reason: str) -> None:
        self._require_revision()
        if self.running_pid():
            raise SystemExit('a session is running; stop it first')
        lock = self.lock
        current = self.fingerprinter()
        validate_pinned_environment(current)
        changes = differences(lock['environment'], current)
        observed = any(path for name in ('probes', 'profiles', 'smoke', 'waits', 'dev', 'eval', 'sensitivity')
                       for path in (self.root / name).rglob('*.json*'))
        if changes and observed:
            raise SystemExit('observations or calibration already exist; use a new campaign for changed '
                             'code or inputs instead of reusing old profiles or partial runs')
        self._append('lock-history.jsonl', dict(time=dt.datetime.now().isoformat(), reason=reason,
                                                changes=changes))
        lock.update(environment=current, git=git_state())
        (self.state / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
        self.note(f'environment relocked: {reason}')

    def _require_revision(self) -> None:
        if self.config.get('experiment_revision') != EXPERIMENT_REVISION:
            raise SystemExit('this directory is a legacy pilot; create a new campaign for the corrected '
                             f'revision {EXPERIMENT_REVISION} instead of resuming or relocking it')

    def running_pid(self) -> int | None:
        path = self.state / 'campaign.pid'
        if not path.exists():
            return None
        pid = int(path.read_text().split()[0])
        try:
            process = psutil.Process(pid)
            if 'phase2.campaign' in ' '.join(process.cmdline()):
                return pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return None

    # ----- stages -----------------------------------------------------------
    def _py(self, module: str, *args) -> list[str]:
        return [self.python, '-m', module, *[str(a) for a in args]]

    def _frozen_labels(self) -> list[str]:
        path = self.root / 'manifests/frozen.json'
        return [item['label'] for item in json.loads(path.read_text())['policies']] if path.exists() else []

    def _key_labels(self, extra=()) -> list[str]:
        path = self.root / 'manifests/frozen.json'
        final = json.loads(path.read_text()).get('final_label', 'joint') if path.exists() else 'joint'
        frozen = self._frozen_labels()
        return list(dict.fromkeys(label for label in [*KEY_LABELS, final, *extra] if label in frozen))

    def _screen_labels(self, config: dict | None = None) -> list[str]:
        config = config or self.config
        return (['fixed-k1-j1', 'fixed-k1-jinf', 'fixed-k2-j2', 'fixed-k2-j3', 'fixed-k2-j4',
                 'fifo', 'affinity', 'affinity-fixed-j2', 'affinity-fixed-j3', 'ram-cache']
                + [f'joint-a{float(aging):g}'.replace('.', 'p')
                   for aging in config['aging_grid'].split(',')])

    def _sensitivity_labels(self) -> list[str]:
        # One strong fixed baseline: the affinity-aware cap when freeze kept it.
        frozen = self._frozen_labels()
        strong = 'affinity-fixed-best' if 'affinity-fixed-best' in frozen else 'best-fixed'
        return [label for label in self._key_labels(SENSITIVITY_EXTRA) if label not in
                {'best-fixed', 'affinity-fixed-best'} - {strong}]

    def _short_cutoff(self, frozen: bool) -> float:
        if frozen:
            return float(json.loads((self.root / 'manifests/frozen.json').read_text())['short_cutoff'])
        tasks = json.loads(DEV_WORKLOAD.read_text())['tasks']
        return float(statistics.median(task_work(task) for task in tasks))

    def stages(self) -> list[Stage]:
        c, r = self.config, self.root
        common = ['--platform-id', c['platform_id']]
        profile, probe_profile = r / 'profiles/profile.json', r / 'profiles/probe.json'
        stages = [Stage(f'probe-{suite}', 'once', 'normal', minutes, fresh=r / f'probes/{suite}/raw',
                        command=lambda suite=suite: self._py('phase2.probes', '--suite', suite,
                            '--results-dir', r / f'probes/{suite}', *common))
                  for suite, minutes in PROBES]
        inputs = [arg for suite, _ in PROBES for arg in ('--input', r / f'probes/{suite}/raw/probes.jsonl')]
        stages.append(Stage('calibrate-probes', 'once', output=probe_profile,
            command=lambda: self._py('phase2.probes', 'calibrate', *inputs, '--output', probe_profile)))
        bench = ['--cost-profile', probe_profile, *common, '--gap', 'lognormal:2:1',
                 '--arrival', c['arrival'], '--skip-done']
        stages.append(Stage('smoke', 'once', 'normal', 10, command=lambda: self._py(
            'phase2.bench', '--workload', DEV_WORKLOAD, '--results-dir', r / 'smoke',
            '--policies', 'fifo,joint,joint-checkpoint', '--seeds', c['seeds']['smoke'],
            '--num-jobs', 2, *bench)))
        stages.append(Stage('collect-waits', 'once', 'normal', 10 * c['minutes_per_task'] * 1.2,
            command=lambda: self._py('phase2.bench', '--workload', DEV_WORKLOAD, '--results-dir',
                r / 'waits', '--policies', 'fifo', '--seeds', c['seeds']['waits'], '--num-jobs', 10, *bench)))
        stages.append(Stage('calibrate', 'once', output=profile, command=lambda: self._py(
            'phase2.probes', 'calibrate', *inputs, '--waits-input',
            r / 'waits/raw/policy_decisions.jsonl', '--output', profile)))

        def create(phase, manifest, results, seeds, *extra):
            return lambda: self._py('phase2.plan', 'create', '--phase', phase, '--workload',
                EVAL_WORKLOAD if phase != 'dev' else DEV_WORKLOAD, '--cost-profile', profile, *common,
                '--results-dir', results, '--manifest', manifest, '--seeds', seeds,
                '--arrival', c['arrival'], *extra)

        def runs(name, manifest, jobs, expected):
            minutes = jobs * c['minutes_per_task'] * 1.2
            return [Stage(name, 'manifest', 'normal', minutes, manifest=manifest,
                          group=SERVERS['normal']['group'], expected_runs=expected),
                    Stage(name + '-ram', 'manifest', 'ram', minutes + 1, manifest=manifest,
                          group=SERVERS['ram']['group'], restart='ram-cache', expected_runs=3)]

        def summary(name, results, baselines, frozen, *extra):
            def command():
                args = ['--results-dir', results, '--short-cutoff', self._short_cutoff(frozen), *extra]
                for baseline in baselines():
                    args += ['--baseline', baseline]
                return self._py('phase2.summarize', *args, '--plot')
            return Stage(name, 'once', command=command)

        m = r / 'manifests'
        stages.append(Stage('dev-manifest', 'once', output=m / 'dev.json', command=create(
            'dev', m / 'dev.json', r / 'dev', c['seeds']['dev'], '--trials', 1,
            '--aging-grid', c['aging_grid'], '--only-labels', ','.join(self._screen_labels()))))
        screen_runs = (len(self._screen_labels()) - 1) * len(c['seeds']['dev'].split(','))
        dev = runs('dev-runs', m / 'dev.json', 10, screen_runs)
        # Gate F: show that the host cache really saves and restores evicted slots.
        ram_probe = Stage('probe-ram-cache', 'once', 'ram', 15, fresh=r / 'probes/ram-cache/raw',
                          command=lambda: self._py('phase2.probes', '--suite', 'ram-cache',
                                                   '--results-dir', r / 'probes/ram-cache', *common))
        stages += [dev[0], ram_probe, dev[1]]
        stages.append(summary('dev-screen-summary', r / 'dev', lambda: ['fifo'], False))

        def confirm():
            shared = select_shared(r / 'dev')
            aging = f'{shared["aging_s"]:g}'
            suffix = aging.replace('.', 'p')
            queue = 'attained' if shared['queue_rule'] == 'attained' else 'queue'
            labels = [f'{name}-a{suffix}' for name in
                      [queue, 'cache', 'joint-checkpoint', 'ttl-10', 'ttl-15', 'ttl-30']]
            return create('dev', m / 'dev-confirm.json', r / 'dev', c['seeds']['dev'], '--trials', 1,
                          '--aging-grid', aging, '--only-labels', ','.join(labels))()

        stages.append(Stage('dev-confirm-manifest', 'once', output=m / 'dev-confirm.json', command=confirm))
        stages.append(Stage('dev-confirm-runs', 'manifest', 'normal', 10 * c['minutes_per_task'] * 1.2,
                            manifest=m / 'dev-confirm.json', group=SERVERS['normal']['group'],
                            expected_runs=6 * len(c['seeds']['dev'].split(','))))
        stages.append(summary('dev-summary', r / 'dev', lambda: ['fifo'], False))
        stages.append(Stage('freeze', 'once', output=m / 'frozen.json', command=lambda: self._py(
            'phase2.freeze', '--dev-results', r / 'dev', '--dev-workload', DEV_WORKLOAD,
            '--checkpoint-budget-gib', c['checkpoint_gib'], '--output', m / 'frozen.json')))
        stages.append(Stage('freeze-review', 'gate'))
        stages.append(Stage('eval-manifest', 'once', output=m / 'eval.json', command=create(
            'eval', m / 'eval.json', r / 'eval', c['seeds']['eval'], '--trials', 1,
            '--frozen-config', m / 'frozen.json')))
        normal_eval = len([label for label in self._frozen_labels() if label != 'ram-cache']) or 9
        stages += runs('eval-runs', m / 'eval.json', 20,
                       normal_eval * len(c['seeds']['eval'].split(',')) + 1)
        key = self._key_labels
        stages.append(Stage('eval-trials-manifest', 'once', output=m / 'eval-trials.json',
            command=lambda: create('eval', m / 'eval-trials.json', r / 'eval', c['seeds']['eval'],
                '--trials', CONFIRM_TRIALS, '--trial-labels', ','.join(key()), '--only-labels', ','.join(key()),
                '--frozen-config', m / 'frozen.json')()))
        stages.append(Stage('eval-trials', 'manifest', 'normal', 20 * c['minutes_per_task'] * 1.2,
                            manifest=m / 'eval-trials.json', group=SERVERS['normal']['group'],
                            expected_runs=(len(key()) or 4) * len(c['seeds']['eval'].split(','))
                                * (CONFIRM_TRIALS - 1)))
        baselines = lambda: [b for b in ('fifo', 'best-fixed', 'affinity-fixed-best', 'affinity')
                             if b in self._frozen_labels()]
        confirm_trials = ['--confirmatory-trials', ','.join(str(t) for t in range(CONFIRM_TRIALS))]
        stages.append(summary('eval-summary', r / 'eval', baselines, True, *confirm_trials))
        stages.append(Stage('sensitivity-manifest', 'once', output=m / 'sensitivity.json',
            command=lambda: create('sensitivity', m / 'sensitivity.json', r / 'sensitivity',
                c['seeds']['sensitivity'], '--trials', 1,
                '--only-labels', ','.join(self._sensitivity_labels()),
                '--conditions', c.get('sensitivity_conditions', SENSITIVITY), '--no-isolated',
                '--frozen-config', m / 'frozen.json')()))
        stages.append(Stage('sensitivity-runs', 'manifest', 'normal', 20 * c['minutes_per_task'] * 1.3,
                            manifest=m / 'sensitivity.json', group=SERVERS['normal']['group'],
                            expected_runs=(len(self._sensitivity_labels()) or 4)
                                * len(c.get('sensitivity_conditions', SENSITIVITY).split(','))
                                * len(c['seeds']['sensitivity'].split(','))))
        stages.append(summary('sensitivity-summary', r / 'sensitivity', baselines, True,
                              '--reference-dir', r / 'eval', *confirm_trials))
        return stages

    def _selected(self, stage: Stage) -> tuple[dict, list[dict]]:
        manifest = json.loads(stage.manifest.read_text())
        return manifest, [run for run in manifest['runs'] if run['server_group'] == stage.group]

    def progress(self, stage: Stage) -> tuple[int, int] | None:
        if stage.kind != 'manifest' or not stage.manifest.exists():
            return None
        manifest, runs = self._selected(stage)
        return sum(run_done(run, manifest) for run in runs), len(runs)

    def done(self, stage: Stage) -> bool:
        if stage.kind == 'gate':
            frozen = self.root / 'manifests/frozen.json'
            approved = self.state / 'approved-frozen.sha256'
            return frozen.exists() and approved.exists() and approved.read_text().split()[0] == sha256(frozen)
        if stage.kind == 'manifest':
            counts = self.progress(stage)
            return counts is not None and counts[0] == counts[1]
        if stage.output is not None:
            return stage.output.exists()
        return (self.state / 'done' / stage.name).exists()

    def approve(self) -> None:
        frozen = self.root / 'manifests/frozen.json'
        if not frozen.exists():
            raise SystemExit('no frozen config yet; dev has not finished')
        (self.state / 'approved-frozen.sha256').write_text(f'{sha256(frozen)}  {dt.datetime.now().isoformat()}\n')
        self.note('frozen config approved for eval: ' + '; '.join(json.loads(frozen.read_text()).get('notes', [])))

    # ----- processes ----------------------------------------------------------
    def _spawn(self, command: list[str]) -> int:
        command = [str(arg) for arg in command]
        with self.log_path.open('a') as log:
            log.write(f'$ {shlex.join(command)}\n')
            log.flush()
            # Own process group: a stop signal reaches the plan executor and its
            # benchmark together, while llama-server (its own session) is untouched.
            self.child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            try:
                return self.child.wait()
            finally:
                self.child = None

    def interrupt(self, reason: str) -> None:
        if self.abort is None:
            self.abort = reason
            self.note(f'aborting current work: {reason}')
        child = self.child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def _watchdog(self, stop: threading.Event) -> None:
        last = time.time()
        while not stop.wait(WATCH_S):
            now = time.time()
            if now - last > WATCH_S * 4:
                self.interrupt(f'machine was asleep for ~{now - last:.0f} s (lid closed?)')
            state = self.power() or ''
            if 'Battery Power' in state:
                self.interrupt('charger unplugged')
            last = now

    def server(self, kind: str | None) -> bool:
        if kind is None or kind == self.server_kind:
            return True
        self.stop_server()
        spec = SERVERS[kind]
        rc = self.runner(self._py('phase2.server', '--parallel', 2, '--ctx-size', 24576,
            '--cache-ram', spec['cache_ram'], '--cache-idle-slots' if spec['idle'] else '--no-cache-idle-slots',
            '--port', 18080, '--results-dir', self.root / spec['directory']))
        if rc == 0:
            self.server_kind = kind
        return rc == 0

    def stop_server(self) -> None:
        self.runner(self._py('phase2.server', 'stop'))
        self.server_kind = None

    def clean_snapshots(self) -> None:
        """Remove snapshot files minted by this project once no backend can write them."""
        if any('llama-server' in (p.info['name'] or '') for p in psutil.process_iter(['name'])):
            self.note('a llama-server is still running; leaving snapshot files untouched')
            return
        removed = 0
        for spec in SERVERS.values():
            directory = self.root / spec['directory'] / 'cache'
            for path in directory.glob('*') if directory.is_dir() else []:
                if path.is_file() and not path.is_symlink() and path.name.startswith(MINTED):
                    removed += path.stat().st_size
                    path.unlink()
        if removed:
            self.note(f'removed {removed / 2**30:.2f} GiB of snapshots left by an interrupted session')

    # ----- session --------------------------------------------------------------
    def preflight(self) -> list[str]:
        self._require_revision()
        problems = differences(self.lock['environment'], self.fingerprinter())
        if problems:
            raise SystemExit('environment differs from the campaign lock:\n  ' + '\n  '.join(problems) +
                             '\nRestore it; before any probes or measurements, relock may record a '
                             'deliberate change. After observations exist, use a new campaign.')
        if 'Battery Power' in (self.power() or ''):
            raise SystemExit('on battery power; connect the charger before starting')
        free = shutil.disk_usage(self.root.parent if not self.root.exists() else self.root).free / 2**30
        if free < 6:
            raise SystemExit(f'only {free:.1f} GiB disk free; at least 6 GiB is needed')
        warnings = []
        available = psutil.virtual_memory().available / 2**30
        if available < 3:
            raise SystemExit(f'only {available:.1f} GiB memory available; close other applications')
        if available < 6:
            warnings.append(f'only {available:.1f} GiB memory available; close other applications to avoid swap')
        try:
            if 'lowpowermode 1' in subprocess.run(['pmset', '-g'], capture_output=True, text=True).stdout:
                warnings.append('Low Power Mode is on; it throttles the M4 and changes timings')
        except OSError:
            pass
        return warnings

    def run(self, until_unix: float | None = None) -> str:
        if self.running_pid():
            raise SystemExit('a session is already running')
        warnings = self.preflight()
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / 'STOP').unlink(missing_ok=True)
        (self.state / 'campaign.pid').write_text(f'{os.getpid()}\n')
        started = self.clock()
        session = dict(start=dt.datetime.now().isoformat(), deadline=until_unix, warnings=warnings,
                       memory_available_gib=round(psutil.virtual_memory().available / 2**30, 2),
                       swap_used_gib=round(psutil.swap_memory().used / 2**30, 2), finished=[])
        for warning in warnings:
            self.note('WARNING ' + warning)
        deadline_text = dt.datetime.fromtimestamp(until_unix).strftime('%a %H:%M') if until_unix else 'none'
        self.note(f'session started (deadline {deadline_text})')
        handlers = {sig: signal.signal(sig, lambda *_: self.interrupt('operator stop'))
                    for sig in (signal.SIGTERM, signal.SIGINT)} \
            if threading.current_thread() is threading.main_thread() else {}
        watch_stop = threading.Event()
        watcher = threading.Thread(target=self._watchdog, args=(watch_stop,), daemon=True)
        watcher.start()
        reason, retried, warmed = 'complete', set(), False
        try:
            self.stop_server()
            self.clean_snapshots()
            stages = self.stages()
            index = 0
            while index < len(stages):
                stage = stages[index]
                if self.abort:
                    reason = self.abort
                    break
                if self.done(stage):
                    index += 1
                    continue
                if (self.state / 'STOP').exists():
                    reason = 'stop requested'
                    break
                if stage.kind == 'gate':
                    reason = 'waiting for review of manifests/frozen.json'
                    self.note('dev is finished. Review the frozen config, then: '
                              'bash scripts/phase2_night.sh approve')
                    break
                minutes = stage.minutes * self.config['speed']
                warm = stage.kind == 'manifest' and not warmed
                needed = minutes + (WARMUP_MINUTES * self.config['speed'] if warm else 0)
                if until_unix and self.clock() + needed * 60 > until_unix:
                    reason = f'deadline (next step {stage.name} needs ~{needed:.0f} min)'
                    break
                if not self.server(stage.server):
                    reason = f'backend failed to start for {stage.name}'
                    break
                if warm:
                    self._warmup()
                    warmed = True
                    if self.abort:
                        reason = self.abort
                        break
                self.note(f'stage {stage.name}')
                rc = self._execute(stage, minutes, until_unix)
                if self.abort:
                    reason = self.abort
                    break
                if rc != 0:
                    if stage.name in retried:
                        reason = f'{stage.name} failed twice; see campaign.log'
                        break
                    retried.add(stage.name)
                    self.note(f'{stage.name} exited with {rc}; restarting the backend and retrying once')
                    if stage.server:
                        self.stop_server()
                    continue
                if stage.kind == 'manifest' and not self.done(stage):
                    reason = 'stop requested' if (self.state / 'STOP').exists() else 'deadline'
                    break
                if stage.kind == 'once' and stage.output is None:
                    (self.state / 'done').mkdir(exist_ok=True)
                    (self.state / 'done' / stage.name).write_text(dt.datetime.now().isoformat() + '\n')
                session['finished'].append(stage.name)
                index += 1
        except BaseException as error:
            # Never record an internal failure as a completed session.
            reason = f'internal error: {type(error).__name__}: {error}'
            raise
        finally:
            watch_stop.set()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            for cleanup in (self.stop_server, self.clean_snapshots):
                try:
                    cleanup()
                except Exception as error:  # still record the session below
                    self.note(f'cleanup {cleanup.__name__} failed: {type(error).__name__}: {error}')
            session.update(end=dt.datetime.now().isoformat(), reason=reason,
                           hours=round((self.clock() - started) / 3600, 2))
            self._append('sessions.jsonl', session)
            self.note(f'session ended: {reason}')
            (self.state / 'campaign.pid').unlink(missing_ok=True)
        return reason

    def _warmup(self) -> None:
        """Unmeasured replay before the session's first measured run (see WARMUP_MINUTES)."""
        profile = self.root / 'profiles/profile.json'
        if not profile.exists():
            profile = self.root / 'profiles/probe.json'
        policy = 'ram-cache' if self.server_kind == 'ram' else 'fifo'
        # One retry: a transient failure should not cost the whole night, but a
        # session never measures without a completed warm-up.
        for attempt in (1, 2):
            target = self.state / 'warmup' / f'{time.strftime("%Y%m%d-%H%M%S")}-{attempt}'
            self.note(f'warm-up: unmeasured replay in {target.relative_to(self.root)}')
            rc = self.runner(self._py(
                'phase2.bench', '--workload', DEV_WORKLOAD, '--results-dir', target, '--policies', policy,
                '--seeds', SEEDS['smoke'], '--num-jobs', 3, '--cost-profile', profile,
                '--platform-id', self.config['platform_id'], '--gap', 'lognormal:2:1',
                '--arrival', self.config['arrival']))
            if rc == 0 or self.abort:
                return
            self.note(f'warm-up attempt {attempt} exited with {rc}')
        self.note('warm-up failed twice; stopping before measured work')
        self.abort = 'warm-up failed'

    def _execute(self, stage: Stage, minutes: float, until_unix: float | None) -> int:
        if stage.kind == 'once':
            if stage.fresh is not None and stage.fresh.exists():
                aside = stage.fresh.with_name(f'raw-interrupted-{time.strftime("%Y%m%d-%H%M%S")}')
                stage.fresh.rename(aside)  # a partial probe suite is redone from the start
                self.note(f'moved partial {stage.fresh} to {aside.name}')
            return self.runner(stage.command())
        command = self._py('phase2.plan', 'run', stage.manifest, '--execute', '--server-group', stage.group,
                           '--budget-s', 10**8, '--stop-file', self.state / 'STOP',
                           '--run-estimate-s', round(minutes * 60), '--restart-server', stage.restart)
        if until_unix:
            command += ['--deadline-unix', f'{until_unix:.0f}']
        return self.runner(command)

    # ----- reporting --------------------------------------------------------------
    def status(self) -> str:
        lines = [f'Campaign {self.root} ({self.config["platform_id"]}, preset {self.config["preset"]})']
        pid = self.running_pid()
        if pid:
            lines.append(f'Session: RUNNING (pid {pid}); stop gracefully with: bash scripts/phase2_night.sh stop')
        sessions = (self.state / 'sessions.jsonl')
        if sessions.exists() and sessions.read_text().strip():
            last = json.loads(sessions.read_text().strip().splitlines()[-1])
            lines.append(f'Last session: {last["start"][:16]} -> {last.get("end", "")[:16]}, '
                         f'{last.get("hours")} h, ended: {last.get("reason")}')
        if self.config.get('experiment_revision') != EXPERIMENT_REVISION:
            lines.append('Legacy pilot: results are preserved; create a new campaign for corrected experiments.')
            return '\n'.join(lines)
        remaining = 0.
        lines.append(f'{"stage":28s} {"state":10s} progress')
        for stage in self.stages():
            counts = self.progress(stage)
            done = self.done(stage)
            state = 'done' if done else ('waiting' if stage.kind == 'gate' else 'pending')
            detail = f'{counts[0]}/{counts[1]} runs' if counts else ''
            if counts and 0 < counts[0] < counts[1]:
                state = 'partial'
            if stage.name.startswith('probe-') and (self.root / f'probes/{stage.name[6:]}/raw/probes.jsonl').exists():
                rows = [json.loads(l) for l in (self.root / f'probes/{stage.name[6:]}/raw/probes.jsonl').read_text().splitlines() if l.strip()]
                bad = sum(row.get('status') != 'ok' for row in rows)
                detail = f'{len(rows)} rows, {bad} not ok' + (' <- review before checkpoint' if bad else '')
            if not done:
                left = (counts[1] - counts[0]) if counts else (1 if stage.kind == 'once' else 0)
                remaining += left * stage.minutes * self.config['speed']
                if stage.kind == 'manifest' and not counts:
                    remaining += stage.minutes * self.config['speed'] * stage.expected_runs
            lines.append(f'{stage.name:28s} {state:10s} {detail}')
        lines.append(f'Rough time left on this platform: ~{remaining / 60:.0f} h of running')
        return '\n'.join(lines)


def root_from(arguments) -> Path:
    if arguments.root:
        return Path(arguments.root)
    if POINTER.exists():
        return Path(POINTER.read_text().strip())
    raise SystemExit('no campaign yet: bash scripts/phase2_night.sh init m4 <platform-id>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', help='campaign directory (default: the one recorded by init)')
    actions = parser.add_subparsers(dest='action', required=True)
    init = actions.add_parser('init', help='lock the environment and create the campaign')
    init.add_argument('preset', choices=sorted(PRESETS))
    init.add_argument('platform_id')
    init.add_argument('--reuse-from', type=Path,
                      help='pilot campaign whose probes, wait samples, profiles and dev runs of '
                           'policies unchanged by this revision are copied (see REUSABLE_POLICIES)')
    run = actions.add_parser('run', help='run stages until done, a deadline, or a stop')
    run.add_argument('--until', help='local HH:MM; no new run starts that would end later')
    run.add_argument('--hours', type=float, help='alternative to --until')
    actions.add_parser('status')
    actions.add_parser('approve', help='approve the frozen dev config for eval')
    relock = actions.add_parser('relock', help='record a deliberate environment change')
    relock.add_argument('reason')
    stop = actions.add_parser('stop')
    stop.add_argument('--now', action='store_true', help='abort the current run (it is redone next time)')
    args = parser.parse_args(argv)
    if args.action == 'init':
        root = Path(args.root) if args.root else PROJECT / 'runs' / f'phase2-{args.platform_id}-r{EXPERIMENT_REVISION}'
        Campaign(root).init(args.preset, args.platform_id, reuse_from=args.reuse_from)
        print(f'Campaign ready at {root}. Start a night with: bash scripts/phase2_night.sh start --until 07:30')
        return
    campaign = Campaign(root_from(args))
    if args.action == 'run':
        campaign.run(parse_until(args.until, args.hours, dt.datetime.now()))
    elif args.action == 'status':
        print(campaign.status())
    elif args.action == 'approve':
        campaign.approve()
    elif args.action == 'relock':
        campaign.relock(args.reason)
    elif args.action == 'stop':
        pid = campaign.running_pid()
        if not pid:
            print('No session is running.')
            return
        if not args.now:
            (campaign.state / 'STOP').write_text(dt.datetime.now().isoformat() + '\n')
            print('Stop requested: the current run or stage finishes first, then the backend stops. '
                  'Use "stop --now" to abort the current run instead.')
            return
        os.kill(pid, signal.SIGTERM)
        print('Aborting the current run; waiting for the backend to drain and stop ...')
        for _ in range(480):
            if not psutil.pid_exists(pid):
                print('Stopped.')
                return
            time.sleep(1)
        print('Still draining; check: bash scripts/phase2_night.sh status')


if __name__ == '__main__':
    main()
