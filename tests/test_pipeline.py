"""scripts/overnight.sh with a stub interpreter: checks which experiments run, in what order."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = '''#!/bin/bash
echo "$*" >> "$STUB_LOG"
[ "$1" = "-c" ] && echo 8   # available GiB for the P4/36K check
[ "$1" = "-" ] && cat > /dev/null
exit 0
'''


def run_pipeline(tmp_path, **env):
    stub = tmp_path / 'python'
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / 'calls.log'
    full_env = dict(os.environ, PY=str(stub), STUB_LOG=str(log), PAUSE='0', HOURS='1',
                    WORKLOADS='runs/final', OUT=str(tmp_path / 'out'), DEV=str(tmp_path / 'dev'), **env)
    subprocess.run(['bash', str(ROOT / 'scripts/overnight.sh')], env=full_env, check=True, timeout=120)
    return log.read_text().splitlines()


def macro_calls(calls):
    return [c for c in calls if c.startswith('-m phase1.bench macro')]


def experiments(calls):
    """(server launch, bench call) pairs: P4, P4u and P4·36K share bench arguments, not servers."""
    server, pairs = None, []
    for c in calls:
        if c.startswith('-m phase1.server --results-dir'):
            server = c.split(' ', 4)[-1]
        elif c.startswith('-m phase1.bench macro'):
            pairs.append((server, c))
    return pairs


def test_trials_rotate_order_and_label_only_later_trials(tmp_path):
    calls = run_pipeline(tmp_path, TRIALS='2', ARRIVAL_KIND='scaled', ARRIVAL_RATE='3')
    macro = macro_calls(calls)
    # Per trial: rounds 0 and 1 (8 P2 cases + 4 other servers), dev 4, round 2 (4 + 4), P4/36K 2.
    assert len(macro) == 2 * (2 * 12 + 4 + 8 + 2)
    assert len(set(experiments(calls))) == len(macro), 'every experiment and trial runs exactly once'
    first, second = [c for c in macro if '--trial' not in c], [c for c in macro if '--trial 2' in c]
    assert len(first) == len(second) == len(macro) // 2
    assert not any('capture' in c for c in calls) and not any('phase1.noise' in c for c in calls)

    round0 = [c for c in macro if '--repeat-start 0 ' in c and 'workload_eval' in c and 'C36864' not in c]
    assert '--k 2 --jobs-active 0' in round0[0]        # trial 1: P2 group first, FIFO first
    t2 = [c for c in round0 if '--trial 2' in c]
    assert '--k 1 --trial 2' in t2[0] and '--jobs-active' not in t2[0]   # trial 2: P1 group first
    p2_t2 = [c for c in t2 if '--jobs-active' in c or '--gap zero' in c or 'fixed:15' in c or 'scaled:1.5' in c or 'scaled:6' in c]
    assert '--k 1 --jobs-active 1' in p2_t2[0]          # P2 cases rotated by one

    assert all('--arrival scaled:' in c for c in macro)
    assert any('--arrival scaled:1.5' in c for c in macro) and any('--arrival scaled:6' in c for c in macro)
    assert sum('--arrival scaled:3 ' in c for c in macro) == len(macro) - 2 * 2 * 2   # minus half/double sweeps
    assert any('phase1.final' in c and '--main-arrival scaled:3' in c for c in calls)


def test_defaults_keep_the_original_labels_and_order(tmp_path):
    macro = macro_calls(run_pipeline(tmp_path, NOISE_TRIALS='0'))
    assert len(macro) == 2 * 12 + 4 + 8 + 2
    assert not any('--trial' in c for c in macro)
    assert {a for c in macro for a in ['poisson:0.5', 'poisson:2'] if f'--arrival {a} ' in c} == {'poisson:0.5', 'poisson:2'}
    assert '--k 2 --jobs-active 0' in macro[0]


def test_extra_admission_limits_join_p2_and_dev(tmp_path):
    macro = macro_calls(run_pipeline(tmp_path, NOISE_TRIALS='0', EXTRA_J='3 4'))
    assert len(macro) == 2 * 14 + 6 + 10 + 2   # rounds 0/1 gain 2 P2 cases, dev 2, round 2 two
    for j in (3, 4):
        assert sum(f'--k 2 --jobs-active {j} ' in c and 'workload_eval' in c for c in macro) == 3
        assert sum(f'--k 2 --jobs-active {j} ' in c and 'workload_dev' in c for c in macro) == 1
