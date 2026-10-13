"""Compare installed TensorFold source against a running container, without loading models."""
import argparse
import json
import subprocess

HASH_CODE = '''import importlib.util,hashlib,json,sys
from pathlib import Path
p=Path(importlib.util.find_spec('tensorfold').origin).parent
normalise = sys.argv[1] == 'normalise'
def digest(path):
    data = path.read_bytes()
    if normalise:
        data = data.replace(b'\\r\\n', b'\\n')
    return hashlib.sha256(data).hexdigest()
print(json.dumps({str(f.relative_to(p)):digest(f)
 for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--image', default='thinkingcap-exl3:0.2.0')
    parser.add_argument('--normalise-line-endings', action='store_true',
                        help='Compare source text after CRLF-to-LF normalisation; not byte equivalence')
    args = parser.parse_args()
    mode = 'normalise' if args.normalise_line_endings else 'exact'
    old = json.loads(subprocess.check_output(['docker', 'exec', args.container, 'python3', '-c', HASH_CODE, mode], text=True))
    new = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'python3',
                                            args.image, '-c', HASH_CODE, mode], text=True))
    mismatches = sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))
    print(json.dumps({'source_files': len(new), 'historical_source_files': len(old),
                      'mismatches': mismatches, 'hash_algorithm': 'SHA-256',
                      'comparison_mode': mode}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
