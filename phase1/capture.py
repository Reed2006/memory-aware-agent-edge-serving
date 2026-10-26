"""Run the unmodified tau Retail orchestrator with a local agent and DeepSeek user.

Only the model transport is replaced. No API key enters tau configuration or logs.
The environment evaluator is programmatic; its score is not the full leaderboard score.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['LITELLM_TELEMETRY'] = 'False'

import httpx
from loguru import logger

logger.remove()
logger.add(lambda msg: None, level='CRITICAL')

from tau2 import TextRunConfig
from tau2.agent import llm_agent
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.domains.retail.environment import get_tasks
from tau2.evaluator.evaluator import EvaluationType
from tau2.environment.environment import Environment
from tau2.runner import run_single_task
from tau2.user import user_simulator
from tau2.utils.llm_utils import to_litellm_messages

from phase1.core import Jsonl, add_results_arg, server_info
from phase1.adapters import normalize_tool_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=291)
    parser.add_argument('--max-steps', type=int, default=40)
    parser.add_argument('--user-model', default='deepseek-flash')
    add_results_arg(parser)
    args = parser.parse_args()
    local = f'http://127.0.0.1:{server_info().get("port", 18080)}'
    output = Path(args.results_dir) / 'raw' / f'live_{args.split}'
    output.mkdir(parents=True, exist_ok=True)
    key = os.environ.get('DEEPSEEK_API_KEY') or Path('API.md').read_text().strip()
    selected = random.Random(args.seed).sample(get_tasks(args.split), args.count)
    (output / 'selection.json').write_text(json.dumps({
        'split': args.split, 'seed': args.seed, 'task_ids': [t.id for t in selected],
        'max_steps': args.max_steps, 'user_model': args.user_model,
        'evaluation': 'ENV only (programmatic database and environment assertions)',
    }, indent=2))
    usage_log = Jsonl(output / 'remote_usage.jsonl')
    client = httpx.Client(timeout=240)
    current: dict = {}
    original_tool_response = Environment.get_response

    def measured_tool_response(environment, message):
        started = time.perf_counter()
        result = original_tool_response(environment, message)
        current.setdefault('tool_events', []).append({
            'name': message.name, 'elapsed_s': time.perf_counter() - started,
            'error': result.error,
        })
        return result

    Environment.get_response = measured_tool_response

    def generate(model, messages, tools=None, tool_choice=None, call_name=None, **kwargs):
        remote = model == args.user_model
        body = {
            'model': model,
            'messages': to_litellm_messages(messages),
            'temperature': 0, 'max_tokens': 256 if remote else 512,
            'stream': False,
        }
        # Explicit nulls in assistant tool_calls are accepted locally but omitted
        # for the remote OpenAI-compatible endpoint.
        for message in body['messages']:
            if message.get('tool_calls', 'absent') is None:
                message.pop('tool_calls')
        if tools:
            body.update(tools=[t.openai_schema for t in tools], tool_choice='auto',
                        parallel_tool_calls=False)
        if remote:
            body['thinking'] = {'type': 'disabled'}
            url = 'https://api.deepseek.com/chat/completions'
            headers = {'Authorization': 'Bearer ' + key}
        else:
            body['chat_template_kwargs'] = {'enable_thinking': False}
            body['cache_prompt'] = True
            url, headers = local + '/v1/chat/completions', {}
        started = time.perf_counter()
        response = client.post(url, json=body, headers=headers)
        if response.status_code != 200:
            # Do not include remote request, headers, or error bodies in errors.
            raise RuntimeError(f'{"remote" if remote else "local"} HTTP {response.status_code}')
        data = response.json()
        elapsed = time.perf_counter() - started
        if remote:
            usage_log.write({'task_id': current['task_id'], 'model': data.get('model'),
                             'usage': data.get('usage'), 'elapsed_s': elapsed})
        else:
            template = client.post(local + '/apply-template', json=body)
            template.raise_for_status()
            prompt = template.json()['prompt']
            token_response = client.post(local + '/tokenize', json={
                'content': prompt, 'add_special': False, 'parse_special': True})
            token_response.raise_for_status()
            tokens = token_response.json()['tokens']
            now = time.perf_counter()
            current['calls'].append({
                'messages': body['messages'], 'tools': body.get('tools'),
                'prompt_tokens': tokens, 'response': data,
                'started_s': started, 'done_s': started + elapsed,
                'between_calls_s': max(0, started - current.get('previous_done', started)),
            })
            current['previous_done'] = now
        choice, converted = normalize_tool_message(
            data['choices'][0]['message'], {t.name for t in tools or []})
        if not remote:
            current['calls'][-1]['normalized_tool_json'] = converted
        calls = [ToolCall(id=t['id'], name=t['function']['name'],
                          arguments=json.loads(t['function']['arguments']))
                 for t in choice.get('tool_calls') or []]
        return AssistantMessage(role='assistant', content=choice.get('content'),
                                tool_calls=calls or None, usage=data.get('usage'),
                                raw_data=data, generation_time_seconds=elapsed)

    llm_agent.generate = generate
    user_simulator.generate = generate
    config = TextRunConfig(
        domain='retail', llm_agent='phase1-qwen3-4b', llm_user=args.user_model,
        max_steps=args.max_steps, max_errors=3, timeout=600,
        max_concurrency=1, hallucination_retries=0, auto_review=False,
    )
    for task in selected:
        target = output / f'task_{task.id}.json'
        if target.exists():
            print('already captured', task.id, flush=True)
            continue
        current.clear()
        current.update(task_id=task.id, calls=[])
        started = time.perf_counter()
        record = {'task_id': task.id, 'split': args.split,
                  'reward_basis': [str(x) for x in task.evaluation_criteria.reward_basis]}
        try:
            simulation = run_single_task(config, task, seed=args.seed,
                                         evaluation_type=EvaluationType.ENV)
            record.update(status='ok', simulation=simulation.model_dump(mode='json'),
                          env_reward=simulation.reward_info.reward)
        except Exception as error:
            # Only our own sanitized transport errors are safe to print.
            record.update(status='error', error_type=type(error).__name__)
            if isinstance(error, RuntimeError) and str(error).startswith(('local HTTP', 'remote HTTP')):
                record['error'] = str(error)
        record.update(elapsed_s=time.perf_counter() - started, calls=current['calls'],
                      tool_events=current.get('tool_events', []))
        target.write_text(json.dumps(record, ensure_ascii=False))
        print(json.dumps({k: v for k, v in record.items() if k not in ['calls', 'simulation']}),
              'agent_calls=', len(current['calls']), flush=True)
    client.close()


if __name__ == '__main__':
    main()
