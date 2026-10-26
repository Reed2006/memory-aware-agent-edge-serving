"""Start or stop the pinned local backend (Metal by default, --cpu on Linux).

The KV budget is --ctx-size tokens in total. With partitioned KV (default) each of
the --parallel slots owns ctx-size / parallel tokens; --kv-unified lets slots share
one pool. Defaults reproduce the original Mac pilot (P=2, 24,576 tokens).
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
import psutil

POINTER = Path('.server.json')


def assert_port_free(port: int) -> None:
    with socket.socket() as reservation:
        try:
            reservation.bind(('127.0.0.1', port))
        except OSError as error:
            raise RuntimeError(f'Local port {port} is already in use; identify its owner before starting another server.') from error


def build_command(model: Path, args, slot_save_path: Path) -> list[str]:
    command = [
        'llama-server', '-m', str(model), '--host', '127.0.0.1', '--port', str(args.port),
        '--alias', 'phase1-qwen3-4b', '--parallel', str(args.parallel), '--ctx-size', str(args.ctx_size),
        '--kv-unified' if args.kv_unified else '--no-kv-unified',
        '--cache-ram', str(args.cache_ram), '--no-cache-idle-slots',
        '--cache-prompt', '--no-context-shift', '--metrics', '--slots',
        '--slot-save-path', str(slot_save_path), '--fit', 'off',
        '--n-gpu-layers', '0' if args.cpu else 'all', '--jinja',
        '--chat-template-kwargs', '{"enable_thinking":false}',
        '--threads', str(args.threads), '--threads-batch', str(args.threads),
        '--batch-size', '512', '--ubatch-size', '256', '--flash-attn', 'on',
        '--log-timestamps',
    ]
    if args.cpu:
        command += ['--no-kv-offload', '--no-op-offload']
    return command


def stop() -> None:
    if not POINTER.exists():
        print('No server recorded in', POINTER)
        return
    info = json.loads(POINTER.read_text())
    try:
        process = psutil.Process(info['pid'])
        # Only signal the process this launcher started, never a reused PID.
        if 'llama-server' not in ' '.join(process.cmdline()):
            raise psutil.NoSuchProcess(info['pid'])
    except psutil.NoSuchProcess:
        print('Recorded server is not running')
        POINTER.unlink()
        return
    os.kill(info['pid'], signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except psutil.TimeoutExpired:
        process.kill()
    POINTER.unlink()
    print('Stopped llama-server PID', info['pid'])


def start(args) -> None:
    root = Path.cwd()
    model = root / 'models/Qwen3-4B-Q4_K_M.gguf'
    if not model.exists():
        raise SystemExit('Download the model recorded in results/manifest.json first.')
    assert_port_free(args.port)
    results = root / args.results_dir
    log_dir, slot_save_path = results / 'logs', results / 'cache'
    log_dir.mkdir(parents=True, exist_ok=True)
    slot_save_path.mkdir(parents=True, exist_ok=True)
    command = build_command(model, args, slot_save_path)
    log_path = log_dir / f'server-P{args.parallel}-C{args.ctx_size}-{int(time.time())}.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    info = {
        'pid': process.pid, 'command': command, 'started_unix': time.time(),
        'port': args.port, 'parallel': args.parallel, 'ctx_size': args.ctx_size,
        'ctx_per_slot': args.ctx_size if args.kv_unified else args.ctx_size // args.parallel,
        'kv_unified': args.kv_unified, 'cache_ram_mib': args.cache_ram, 'cpu': args.cpu,
        'slot_save_path': str(slot_save_path), 'log': str(log_path),
    }
    (results / 'server.json').write_text(json.dumps(info, indent=2))
    POINTER.write_text(json.dumps(info, indent=2))
    for _ in range(240):
        if process.poll() is not None:
            POINTER.unlink(missing_ok=True)
            raise SystemExit(f'llama-server exited during startup; inspect {log_path}')
        try:
            response = httpx.get(f'http://127.0.0.1:{args.port}/health', timeout=1)
            if response.status_code == 200 and response.json().get('status') == 'ok':
                print('llama-server ready, PID', process.pid, json.dumps({
                    k: info[k] for k in ['parallel', 'ctx_size', 'ctx_per_slot', 'kv_unified', 'cache_ram_mib']}))
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(.5)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    POINTER.unlink(missing_ok=True)
    raise SystemExit('llama-server startup timed out; its process was stopped.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', choices=['start', 'stop'], default='start')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=2, help='Server slots P')
    parser.add_argument('--ctx-size', type=int, default=24576, help='Total KV budget in tokens')
    parser.add_argument('--kv-unified', action='store_true', help='Share one KV pool across slots')
    parser.add_argument('--cache-ram', type=int, default=0, help='Host prompt cache MiB (llama.cpp built-in)')
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    stop() if args.action == 'stop' else start(args)


if __name__ == '__main__':
    main()
