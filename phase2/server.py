"""Pinned Phase 2 backend launcher; never launched implicitly by a benchmark."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import psutil

from phase1.server import assert_port_free, build_command as phase1_command


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_args(args) -> None:
    if args.parallel < 1 or args.threads < 1 or args.ctx_size < 1 or args.cache_ram < 0:
        raise ValueError('slots, threads and context must be positive; cache RAM nonnegative')
    if args.ctx_size % args.parallel:
        raise ValueError('partitioned context budget must divide evenly among slots')
    if args.kv_unified:
        raise ValueError('Phase 2 uses partitioned KV; unified mode is not comparable')


def build_command(model: Path, args, slot_save_path: Path) -> list[str]:
    validate_args(args)
    command = phase1_command(model, args, slot_save_path)
    command[command.index('--alias') + 1] = 'phase2-qwen3-4b'
    if args.cache_idle_slots:
        command[command.index('--no-cache-idle-slots')] = '--cache-idle-slots'
    # Keep f16 KV explicit rather than relying on a future runtime default.
    command += ['--cache-type-k', 'f16', '--cache-type-v', 'f16']
    return command


def stop(record_path=Path('.server.json')) -> None:
    record_path = Path(record_path)
    if not record_path.exists():
        print('No server recorded in', record_path)
        return
    info = json.loads(record_path.read_text())
    try:
        process = psutil.Process(info['pid'])
        if 'llama-server' not in ' '.join(process.cmdline()):
            raise psutil.NoSuchProcess(info['pid'])
        started = info.get('process_create_time')
        if started is not None and abs(process.create_time() - started) > .01:
            raise psutil.NoSuchProcess(info['pid'])
    except psutil.NoSuchProcess:
        record_path.unlink(missing_ok=True)
        print('Recorded server is not running')
        return
    os.kill(info['pid'], signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    record_path.unlink(missing_ok=True)
    print('Stopped llama-server PID', info['pid'])


def start(args) -> dict:
    validate_args(args)
    model = Path(args.model).resolve()
    if not model.is_file():
        raise ValueError('model file does not exist; use the pinned Phase 1 GGUF')
    record = Path(args.server_record).resolve()
    if record.exists():
        info = json.loads(record.read_text())
        if psutil.pid_exists(info.get('pid', -1)):
            raise RuntimeError('a backend is already recorded; stop or inspect it first')
    assert_port_free(args.port)
    results = Path(args.results_dir).resolve()
    log_dir, cache = results / 'logs', results / 'cache'
    log_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    model_hash = file_hash(model)
    command = build_command(model, args, cache)
    log_path = log_dir / f'server-P{args.parallel}-C{args.ctx_size}-{time.time_ns()}.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    info = dict(pid=process.pid, process_create_time=psutil.Process(process.pid).create_time(),
        command=command, started_unix=time.time(), port=args.port, parallel=args.parallel,
        ctx_size=args.ctx_size, ctx_per_slot=args.ctx_size // args.parallel,
        kv_unified=False, cache_ram_mib=args.cache_ram, cache_idle_slots=args.cache_idle_slots,
        cpu=args.cpu, model_path=str(model), model_hash=model_hash, kv_precision='f16',
        slot_save_path=str(cache), log=str(log_path), schema_version=2)
    (results / 'server.json').write_text(json.dumps(info, indent=2))
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(info, indent=2))
    try:
        for _ in range(240):
            if process.poll() is not None:
                raise RuntimeError('backend exited during startup; inspect its log')
            try:
                response = httpx.get(f'http://127.0.0.1:{args.port}/health', timeout=1)
                if response.status_code == 200 and response.json().get('status') == 'ok':
                    print('llama-server ready, PID', process.pid)
                    return info
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(.5)
        raise TimeoutError('backend startup timed out')
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record.unlink(missing_ok=True)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', choices=['start', 'stop'], default='start')
    p.add_argument('--model', type=Path, default=Path('models/Qwen3-4B-Q4_K_M.gguf'))
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--ctx-size', type=int, default=24576)
    p.add_argument('--kv-unified', action='store_true')
    p.add_argument('--cache-ram', type=int, default=0)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--cache-idle-slots', action='store_true', default=False)
    group.add_argument('--no-cache-idle-slots', action='store_false', dest='cache_idle_slots')
    p.add_argument('--port', type=int, default=18080)
    p.add_argument('--results-dir', type=Path, default=Path('runs/phase2'))
    p.add_argument('--server-record', type=Path, default=Path('.server.json'))
    return p


def main():
    args = parser().parse_args()
    stop(args.server_record) if args.action == 'stop' else start(args)


if __name__ == '__main__':
    main()
