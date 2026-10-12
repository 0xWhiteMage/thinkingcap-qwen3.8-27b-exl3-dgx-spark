"""Small public API probe; records real tokens, errors and cap hits, not benchmark claims."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]


def parse_stream(lines):
    started = time.perf_counter()
    first = None
    content, reasoning = [], []
    usage = None
    finish = None
    done = False
    for line in lines:
        text = line.decode('utf-8').strip()
        if not text.startswith('data:'):
            continue
        payload = text[5:].strip()
        if payload == '[DONE]':
            done = True
            break
        event = json.loads(payload)
        if event.get('error'):
            raise ValueError('Server error: ' + json.dumps(event['error']))
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices', []):
            delta = choice.get('delta', {})
            thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
            answer = delta.get('content') or ''
            if first is None and (thought or answer):
                first = time.perf_counter() - started
            content.append(answer)
            reasoning.append(thought)
            finish = choice.get('finish_reason') or finish
    if not done or finish is None or first is None:
        raise ValueError('Incomplete generation stream')
    if not usage or not isinstance(usage.get('completion_tokens'), int) or usage['completion_tokens'] <= 0:
        raise ValueError('Missing/invalid token usage; cannot report throughput')
    return {'content': ''.join(content), 'reasoning_content': ''.join(reasoning),
            'usage': usage, 'finish_reason': finish, 'stream_first_text_s': first}


def request(endpoint, model, prompt, max_tokens, thinking, greedy):
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
               'stream': True, 'stream_options': {'include_usage': True},
               'max_tokens': max_tokens, 'temperature': 0.0 if greedy else 1.0,
               'top_p': 0.95, 'top_k': 20,
               'chat_template_kwargs': {'enable_thinking': thinking, 'reasoning_effort': 'xhigh'}}
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('RECIPE_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['RECIPE_API_KEY']
    started = time.perf_counter()
    req = urllib.request.Request(endpoint.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=1800) as response:
        header_time = time.perf_counter() - started
        result = parse_stream(response)
    result['ttft_s'] = header_time + result.pop('stream_first_text_s')
    result['wall_s'] = time.perf_counter() - started
    result['capped'] = result['finish_reason'] == 'length'
    result['has_final_answer'] = bool(result['content'].strip())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8888')
    parser.add_argument('--model', default='Qwen3.8 27B Kearuga (ThinkingCap)')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--requests', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--thinking', choices=['on', 'off'], default='on')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.concurrency <= 16 and args.requests >= args.concurrency and args.max_tokens > 0):
        parser.error('Require 1..16 concurrency, requests >= concurrency and positive max-tokens')
    if args.output.exists():
        parser.error('Output already exists; choose a new receipt path')
    prompts = json.loads((ROOT / 'bench' / 'prompts.json').read_text())

    def one(index):
        entry = prompts[index % len(prompts)]
        try:
            data = request(args.endpoint, args.model, entry['prompt'], args.max_tokens,
                           args.thinking == 'on', args.greedy)
            return {'index': index, 'id': entry['id'], 'status': 'ok', **data}
        except (OSError, ValueError, KeyError) as error:
            return {'index': index, 'id': entry['id'], 'status': 'error', 'error': str(error)}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        records = list(executor.map(one, range(args.requests)))
    wall = time.perf_counter() - started
    errors = [item for item in records if item['status'] != 'ok']
    summary = {'requests': len(records), 'errors': len(errors), 'wall_s': wall,
               'aggregate_completion_tok_s': None if errors else sum(item['usage']['completion_tokens'] for item in records) / wall,
               'cap_hits': sum(item.get('capped', False) for item in records)}
    report = {'schema': 'public-api-probe-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Small public prompt probe, NOT the historical internal performance or fidelity battery',
              'settings': {key: value for key, value in vars(args).items() if key not in ['output', 'endpoint']},
              'summary': summary, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
