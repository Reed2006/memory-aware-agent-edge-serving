"""Start the pinned local backend, or reuse the same command with --cpu on Linux."""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import httpx


def assert_port_free(port: int) -> None:
    with socket.socket() as reservation:
        try:
            reservation.bind(('127.0.0.1', port))
        except OSError as error:
            raise RuntimeError(f'Local port {port} is already in use; identify its owner before starting another server.') from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--ctx-size', type=int, default=24576)
    parser.add_argument('--cache-ram', type=int, default=0)
    args = parser.parse_args()
    root = Path.cwd()
    model = root / 'models/Qwen3-4B-Q4_K_M.gguf'
    if not model.exists():
        raise SystemExit('Download the model recorded in results/manifest.json first.')
    assert_port_free(18080)
    log_dir = root / 'results/logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    (root / 'results/cache').mkdir(parents=True, exist_ok=True)
    command = [
        'llama-server', '-m', str(model), '--host', '127.0.0.1', '--port', '18080',
        '--alias', 'phase1-qwen3-4b', '--parallel', '2', '--ctx-size', str(args.ctx_size),
        '--no-kv-unified', '--cache-ram', str(args.cache_ram), '--no-cache-idle-slots',
        '--cache-prompt', '--no-context-shift', '--metrics', '--slots',
        '--slot-save-path', str(root / 'results/cache'), '--fit', 'off',
        '--n-gpu-layers', '0' if args.cpu else 'all', '--jinja',
        '--chat-template-kwargs', '{"enable_thinking":false}',
        '--threads', str(args.threads), '--threads-batch', str(args.threads),
        '--batch-size', '512', '--ubatch-size', '256', '--flash-attn', 'on',
        '--log-timestamps',
    ]
    if args.cpu:
        command += ['--no-kv-offload', '--no-op-offload']
    with (log_dir / f'server-{int(time.time())}.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    Path('results/server.json').write_text(json.dumps({
        'pid': process.pid, 'command': command, 'started_unix': time.time(),
    }, indent=2))
    for _ in range(120):
        if process.poll() is not None:
            raise SystemExit('llama-server exited during startup; inspect results/logs/server-*.log')
        try:
            response = httpx.get('http://127.0.0.1:18080/health', timeout=1)
            if response.status_code == 200 and response.json().get('status') == 'ok':
                print('llama-server ready, PID', process.pid)
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
    raise SystemExit('llama-server startup timed out; its process was stopped.')


if __name__ == '__main__':
    main()
