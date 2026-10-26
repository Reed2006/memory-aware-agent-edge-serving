"""Restart llama-server for each server-side configuration, then run one bench suite.

Example (everything after -- goes to phase1.bench):
  python -m phase1.sweep --results-dir runs/p-sweep --parallel 1,2,4 --ctx-size 24576 \
      -- macro --k 1,2,4 --jobs-active 0,2 --gap fixed:2 --arrival poisson:0.5 --num-jobs 8
Server options that do not vary (e.g. --cpu --threads 8) go in --server-args.
"""

import argparse
import itertools
import shlex
import subprocess
import sys


def run(command: list[str]) -> None:
    print('+', shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    if '--' not in sys.argv:
        raise SystemExit('Separate bench arguments with --')
    split = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--parallel', default='2')
    parser.add_argument('--ctx-size', default='24576')
    parser.add_argument('--cache-ram', default='0')
    parser.add_argument('--kv-unified', default='0', help='Comma list of 0/1')
    parser.add_argument('--server-args', default='', help='Extra fixed phase1.server arguments')
    args = parser.parse_args(sys.argv[1:split])
    bench_args = sys.argv[split + 1:]
    python = sys.executable
    grid = itertools.product(args.parallel.split(','), args.ctx_size.split(','),
                             args.cache_ram.split(','), args.kv_unified.split(','))
    for parallel, ctx, cache_ram, unified in grid:
        run([python, '-m', 'phase1.server', 'stop'])
        server = [python, '-m', 'phase1.server', '--results-dir', args.results_dir,
                  '--parallel', parallel, '--ctx-size', ctx, '--cache-ram', cache_ram,
                  *shlex.split(args.server_args)]
        if unified == '1':
            server.append('--kv-unified')
        run(server)
        try:
            run([python, '-m', 'phase1.bench', *bench_args, '--results-dir', args.results_dir])
        finally:
            run([python, '-m', 'phase1.server', 'stop'])


if __name__ == '__main__':
    main()
