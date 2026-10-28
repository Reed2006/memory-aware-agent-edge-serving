"""Server startup cost parsed from llama-server logs (written with --log-timestamps).

Each log line starts with minutes.seconds.milliseconds.microseconds since the server
started logging. One row per launch: model load (`loading model` -> `model loaded`),
the time until the HTTP server listens, and the first request after start. For bench
launches that first request is the 16-token warm-up, so it shows the cold compute
path; later requests run warm. Standard library only, so it runs without the venv.
"""

import argparse
import csv
import re
from pathlib import Path

STAMP = re.compile(r'(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ')
NAME = re.compile(r'server-P(\d+)-C(\d+)-(\d+)\.log$')
SLOTS = re.compile(r"n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'")
TOTAL = re.compile(r'\|\s+total time =\s+([\d.]+) ms /\s+(\d+) tokens')
FIELDS = ['log', 'P', 'ctx_size', 'launched_unix', 'kv_unified', 'ctx_per_slot',
          'load_s', 'ready_s', 'first_request_ms', 'first_request_tokens']


def stamp_s(line: str) -> float | None:
    match = STAMP.match(line)
    if not match:
        return None
    minutes, seconds, ms, us = map(int, match.groups())
    return minutes * 60 + seconds + ms / 1e3 + us / 1e6


def parse_log(path: Path) -> dict:
    row = dict.fromkeys(FIELDS)
    row['log'] = path.name
    if name := NAME.search(path.name):
        row.update(P=int(name[1]), ctx_size=int(name[2]), launched_unix=int(name[3]))
    load_start = None
    for line in path.read_text(errors='replace').splitlines():
        t = stamp_s(line)
        if t is None:
            continue
        if 'load_model: loading model' in line and load_start is None:
            load_start = t
        elif 'llama_server: model loaded' in line and load_start is not None:
            row['load_s'] = t - load_start
        elif 'llama_server: listening on' in line:
            row['ready_s'] = t
        elif slots := SLOTS.search(line):
            row.update(ctx_per_slot=int(slots[2]), kv_unified=slots[3] == 'true')
        elif total := TOTAL.search(line):
            row.update(first_request_ms=float(total[1]), first_request_tokens=int(total[2]))
            break  # every later request runs on a warm server
    return row


def startup_rows(results: Path) -> list[dict]:
    return [parse_log(p) for p in sorted((results / 'logs').glob('server-*.log'))]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', required=True)
    root = Path(parser.parse_args().results_dir)
    rows = startup_rows(root)
    write_csv(rows, root / 'summary/startup.csv')
    print(f"{'log':36s} {'load s':>7s} {'ready s':>8s} {'first request':>18s}")
    for r in rows:
        first = f"{r['first_request_ms']:.0f} ms / {r['first_request_tokens']} tok" if r['first_request_ms'] else 'none'
        load = f"{r['load_s']:.2f}" if r['load_s'] is not None else 'n/a'
        ready = f"{r['ready_s']:.2f}" if r['ready_s'] is not None else 'n/a'
        print(f"{r['log']:36s} {load:>7s} {ready:>8s} {first:>18s}")
    print('Wrote', root / 'summary/startup.csv')


if __name__ == '__main__':
    main()
