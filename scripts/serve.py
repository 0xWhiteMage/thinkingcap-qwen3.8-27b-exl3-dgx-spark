"""Safe, explicit Docker launcher; never stops or replaces another service."""
import argparse
import json
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import time
import urllib.request
import urllib.error
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifacts import ROOT, verify

IMAGE = 'thinkingcap-exl3:0.1.0'
MODEL_NAME = 'Qwen3.8 27B Kearuga (ThinkingCap)'


def command(args):
    return ['docker', 'run', '-d', '--name', args.name, '--restart', 'unless-stopped',
            '--gpus', 'all', '--ipc=host', '--network', 'host', '--cpuset-cpus', args.cpuset,
            '-v', f'{args.models_dir.resolve()}/target:/models/target:ro',
            '-v', f'{args.models_dir.resolve()}/drafter:/models/drafter:ro',
            '-v', f'{args.cache_dir.resolve()}:/tensorfold-cache',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
            '-e', 'TORCH_EXTENSIONS_DIR=/tensorfold-cache/torch_extensions',
            '-e', 'TRITON_CACHE_DIR=/tensorfold-cache/triton', args.image,
            'tensorfold', 'serve', '/models/target', '--drafter', '/models/drafter',
            '--backend', 'cuda', '--kv-dtype', 'bf16', '--drafter-bits', '4',
            '--no-prefill-fp8',
            '--parallel', str(args.parallel), '--context', str(args.context),
            '--host', args.host, '--port', str(args.port), '--name', MODEL_NAME,
            '--max-tokens', '131072', '--no-update-check']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache')
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--name', default='thinkingcap-exl3')
    parser.add_argument('--host', choices=['127.0.0.1', '0.0.0.0'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--context', type=int, default=262144)
    parser.add_argument('--cpuset', default='5-9,15-19')
    parser.add_argument('--timeout', type=int, default=2700)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.port <= 65535 and 1 <= args.parallel <= 16 and 1 <= args.context <= 262144
            and args.timeout > 0):
        parser.error('port, parallel, context or timeout outside supported bounds')
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if platform.machine() != 'aarch64':
        raise SystemExit('This recipe is qualified for Linux aarch64 / DGX Spark only.')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    if 'GB10' not in gpu:
        raise SystemExit('This recipe is qualified for an NVIDIA GB10; nothing was started.')
    for role in ['target', 'drafter']:
        manifest = json.loads((ROOT / 'manifests' / (role + '.json')).read_text())
        verify(manifest, args.models_dir / role)
    subprocess.run(['docker', 'image', 'inspect', args.image], check=True, stdout=subprocess.DEVNULL)
    existing = subprocess.run(['docker', 'container', 'inspect', args.name],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if existing.returncode == 0:
        raise SystemExit('Container already exists. Stop/remove it explicitly; nothing was changed.')
    with socket.socket() as listener:
        try:
            listener.bind((args.host, args.port))
        except OSError as error:
            raise SystemExit(f'Port unavailable; nothing was started: {error}') from error
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    endpoint = f'http://127.0.0.1:{args.port}'
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        state = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Status}}', args.name], text=True).strip()
        if state not in ['running', 'created']:
            raise SystemExit('Container exited; inspect docker logs ' + args.name)
        try:
            with urllib.request.urlopen(endpoint + '/v1/models', timeout=5) as response:
                models = json.load(response)['data']
            if MODEL_NAME in [item['id'] for item in models]:
                print('Ready: ' + endpoint + '/v1')
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(5)
    raise SystemExit('Readiness timeout. Container retained for inspection: docker logs ' + args.name)


if __name__ == '__main__':
    main()
