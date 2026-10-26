"""FIFO admission, request measurements, and append-only token replay."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx


class Jsonl:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, value: dict) -> None:
        with self.path.open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')


class ModelClient:
    def __init__(self, base_url: str, transport=None, timeout: float = 300):
        self.http = httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            transport=transport,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def post(self, route: str, payload: dict) -> dict:
        response = await self.http.post(route, json=payload)
        response.raise_for_status()
        return response.json()

    async def complete(self, payload: dict) -> dict:
        return await self.post('/completion', payload)

    async def tokens(self, text: str) -> list[int]:
        return (await self.post('/tokenize', {'content': text, 'add_special': False}))['tokens']

    async def props(self) -> dict:
        response = await self.http.get('/props')
        response.raise_for_status()
        return response.json()

    async def erase(self, slots: int = 2) -> None:
        for slot in range(slots):
            await self.post(f'/slots/{slot}?action=erase', {})


class Fifo:
    """Dispatch ready requests in enqueue order; tool waits occur outside this class.

    A cancelled caller does not cancel an already admitted backend operation. The
    worker waits for it to finish before freeing its capacity, avoiding oversubscription.
    """

    def __init__(self, concurrency: int, operation: Callable[[dict], Awaitable[dict]], log: Jsonl):
        if concurrency < 1:
            raise ValueError('concurrency must be positive')
        self.concurrency, self.operation, self.log = concurrency, operation, log
        self.queue = asyncio.Queue()
        self.sequence = 0
        self.workers: list[asyncio.Task] = []

    async def __aenter__(self):
        self.workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]
        return self

    async def __aexit__(self, *args):
        await self.queue.join()
        for _ in self.workers:
            self.queue.put_nowait(None)
        await asyncio.gather(*self.workers)

    async def submit(self, payload: dict, metadata: dict) -> dict:
        future = asyncio.get_running_loop().create_future()
        sequence = self.sequence
        self.sequence += 1
        self.queue.put_nowait((sequence, time.perf_counter(), payload, metadata, future))
        return await future

    async def _worker(self):
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            sequence, enqueued, payload, metadata, future = item
            started = time.perf_counter()
            row: dict[str, Any] = {
                **metadata, 'sequence': sequence, 'enqueue_s': enqueued,
                'dispatch_s': started, 'queue_s': started - enqueued,
                'input_tokens': len(payload['prompt']) if isinstance(payload.get('prompt'), list) else None,
            }
            try:
                if future.cancelled():
                    row['status'] = 'cancelled_before_dispatch'
                    continue
                response = await self.operation(payload)
                row.update(status='ok', timings=response.get('timings', {}),
                           slot=response.get('id_slot'), tokens_cached=response.get('tokens_cached'),
                           tokens_evaluated=response.get('tokens_evaluated'),
                           truncated=response.get('truncated', False))
                if not future.done():
                    future.set_result(response)
            except Exception as error:
                row.update(status='error', error_type=type(error).__name__)
                if isinstance(error, httpx.HTTPStatusError):
                    # Local llama-server bodies (e.g. context overflow) carry no secrets.
                    row['error'] = f'HTTP {error.response.status_code}: {error.response.text[:300]}'
                if not future.done():
                    future.set_exception(error)
            finally:
                row.update(done_s=time.perf_counter())
                row['roundtrip_s'] = row['done_s'] - started
                self.log.write(row)
                self.queue.task_done()


def add_results_arg(parser, default='results'):
    parser.add_argument('--results-dir', default=default,
                        help='Root for raw logs, summaries and metadata of one experiment')


def server_info(path: Path = Path('.server.json')) -> dict:
    """Metadata written by phase1.server for the backend that is currently running."""
    return json.loads(path.read_text()) if path.exists() else {}


def append_step(previous: list[int], generated: list[int], delta: list[int]) -> list[int]:
    return previous + generated + delta


def common_prefix(left: list[int], right: list[int]) -> int:
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))
