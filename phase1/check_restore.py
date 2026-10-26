"""Independent post-benchmark correctness check for slot state restoration."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from phase1.bench import payload
from phase1.core import ModelClient, add_results_arg, append_step, server_info


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


async def main(results: Path):
    info = server_info()
    slot_dir = Path(info.get('slot_save_path', results / 'cache'))
    original = slot_dir / 'validation-original.bin'
    restored = slot_dir / 'validation-restored.bin'
    async with ModelClient(f'http://127.0.0.1:{info.get("port", 18080)}') as client:
        await client.erase(info.get('parallel', 2))
        tokens = (await client.tokens(
            'An agent retains historical evidence about an order and its items. ' * 900))[:4096]
        first = await client.complete(payload(tokens, 32, id_slot=0))
        delta = await client.tokens('\nTool result: order is eligible. Continue.\n')
        prompt = append_step(tokens, first['tokens'], delta)
        save = await client.post('/slots/0?action=save', {'filename': original.name})
        expected = await client.complete(payload(prompt, 64, id_slot=0))
        await client.post('/slots/0?action=erase', {})
        await client.complete(payload(list(reversed(tokens)), 32, id_slot=0))
        await client.post('/slots/0?action=erase', {})
        restore = await client.post('/slots/0?action=restore', {'filename': original.name})
        save_again = await client.post('/slots/0?action=save', {'filename': restored.name})
        state_hash_equal = digest(original) == digest(restored)
        actual = await client.complete(payload(prompt, 64, id_slot=0))
        assert len(expected['tokens']) == len(actual['tokens']) == 64
        result = dict(
            input_tokens=len(tokens), compared_generated_tokens=64,
            generated_tokens_equal=expected['tokens'] == actual['tokens'],
            serialized_state_hash_equal=state_hash_equal,
            original_state_sha256=digest(original), restored_state_sha256=digest(restored),
            saved_tokens=save['n_saved'], restored_tokens=restore['n_restored'],
            saved_again_tokens=save_again['n_saved'],
            expected_output_tokens=expected['tokens'], restored_output_tokens=actual['tokens'],
            expected_timings=expected['timings'], restored_timings=actual['timings'],
        )
        (results / 'restore_validation.json').write_text(json.dumps(result, indent=2))
        original.unlink(missing_ok=True)
        restored.unlink(missing_ok=True)
        assert result['generated_tokens_equal'], 'Restoration changed the deterministic continuation'
        assert result['saved_tokens'] == result['restored_tokens'] == result['saved_again_tokens']
        print('Restore check: 64 generated tokens identical; serialized state equality:', state_hash_equal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_results_arg(parser)
    asyncio.run(main(Path(parser.parse_args().results_dir)))
