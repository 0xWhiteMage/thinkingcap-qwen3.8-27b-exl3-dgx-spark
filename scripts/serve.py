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

IMAGE = 'thinkingcap-exl3:0.2.0'
MODEL_NAME = 'Qwen3.8 27B Kearuga (ThinkingCap)'


def command(args):
    experimental = args.profile == 'cuda24'
    cmd = ['docker', 'run', '-d', '--name', args.name, '--restart', 'unless-stopped',
           '--gpus', f'device={args.gpu}' if experimental else 'all',
           '--ipc=host', '--network', 'host']
    if args.cpuset:
        cmd += ['--cpuset-cpus', args.cpuset]
    cmd += ['-v', f'{args.models_dir.resolve()}/target:/models/target:ro',
            '-v', f'{args.cache_dir.resolve()}:/tensorfold-cache',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
            '-e', 'TORCH_EXTENSIONS_DIR=/tensorfold-cache/torch_extensions',
            '-e', 'TRITON_CACHE_DIR=/tensorfold-cache/triton']
    if experimental:
        cmd += ['-e', 'TENSORFOLD_MIN_CAPABILITY=8.0',
                '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=2']
        draft = ['--drafter', 'none', '--no-drafts', '--tp', '1']
    else:
        cmd += ['-v', f'{args.models_dir.resolve()}/drafter:/models/drafter:ro']
        draft = ['--drafter', '/models/drafter', '--drafter-bits', '4']
    return cmd + [args.image, 'tensorfold', 'serve', '/models/target', *draft,
                  '--backend', 'cuda', '--kv-dtype', 'bf16', '--no-prefill-fp8',
                  '--parallel', str(args.parallel), '--context', str(args.context),
                  '--host', args.host, '--port', str(args.port), '--name', MODEL_NAME,
                  '--max-tokens', str(args.max_tokens), '--no-update-check']


def preflight_command(args):
    return ['docker', 'run', '--rm', '--gpus', f'device={args.gpu}',
            '--entrypoint', 'python3', '-e', 'TENSORFOLD_MIN_CAPABILITY=8.0',
            '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=2',
            '-v', f'{args.models_dir.resolve()}/target:/models/target:ro',
            '-v', f'{ROOT}/scripts/capacity_check.py:/capacity_check.py:ro',
            args.image, '/capacity_check.py', '--model', '/models/target',
            '--context', str(args.context), '--live']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['spark', 'cuda24'], default='spark')
    parser.add_argument('--experimental-24gb', action='store_true',
                        help='acknowledge the unqualified, tight-headroom 24 GB attempt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache')
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--name', default='thinkingcap-exl3')
    parser.add_argument('--host', choices=['127.0.0.1', '0.0.0.0'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--parallel', type=int)
    parser.add_argument('--context', type=int)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--cpuset')
    parser.add_argument('--timeout', type=int, default=2700)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    experimental = args.profile == 'cuda24'
    args.parallel = args.parallel if args.parallel is not None else (1 if experimental else 16)
    args.context = args.context if args.context is not None else (8192 if experimental else 262144)
    args.max_tokens = args.max_tokens if args.max_tokens is not None else min(
        args.context, 4096 if experimental else 131072)
    if args.cpuset is None and not experimental:
        args.cpuset = '5-9,15-19'
    if not (1 <= args.port <= 65535 and 1 <= args.parallel <= 16 and 1 <= args.context <= 262144
            and 1 <= args.max_tokens <= args.context and args.timeout > 0 and args.gpu >= 0):
        parser.error('port, parallel, context, max-tokens, GPU or timeout outside supported bounds')
    if experimental and (args.parallel != 1 or args.context > 8192 or args.max_tokens > 4096):
        parser.error('experimental cuda24 requires one lane, context <=8192 and max-tokens <=4096')
    return args


def main(argv=None):
    args = parse_args(argv)
    cmd = command(args)
    if args.profile == 'cuda24':
        print('EXPERIMENTAL: header admission is not measured 24 GB support.', flush=True)
        print('Preflight: ' + shlex.join(preflight_command(args)), flush=True)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if args.profile == 'cuda24' and not args.experimental_24gb:
        raise SystemExit('24 GB is not qualified. Read docs/24GB-GPUS.md and explicitly pass --experimental-24gb.')
    if platform.system() != 'Linux':
        raise SystemExit('Run this Docker launcher on Linux or inside WSL2; nothing was started.')
    if args.profile == 'spark':
        if platform.machine() != 'aarch64':
            raise SystemExit('The Spark profile requires Linux aarch64 / DGX Spark.')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
        if 'GB10' not in gpu:
            raise SystemExit('The Spark profile requires NVIDIA GB10; nothing was started.')
    elif platform.machine() != 'x86_64':
        raise SystemExit('The experimental discrete-GPU profile requires Linux x86_64.')
    for role in (['target'] if args.profile == 'cuda24' else ['target', 'drafter']):
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
    if args.profile == 'cuda24':
        subprocess.run(preflight_command(args), check=True)
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
