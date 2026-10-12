"""Download immutable Hugging Face snapshots and verify their actual bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(manifest, directory):
    directory = Path(directory)
    for entry in manifest['files']:
        path = directory / entry['path']
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError('Missing or size-mismatched file: ' + str(path))
        algorithm = 'sha256' if 'sha256' in entry else 'git_blob_sha1'
        digest = hashlib.sha256() if algorithm == 'sha256' else hashlib.sha1()
        if algorithm == 'git_blob_sha1':
            digest.update(f"blob {entry['size']}\0".encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry[algorithm]:
            raise ValueError('Checksum mismatch: ' + str(path))
    return len(manifest['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--role', choices=['target', 'drafter', 'both'], default='both')
    args = parser.parse_args()
    for role in ['target', 'drafter'] if args.role == 'both' else [args.role]:
        manifest = json.loads((ROOT / 'manifests' / (role + '.json')).read_text())
        directory = args.models_dir / role
        if not args.verify_only:
            subprocess.run(['hf', 'download', manifest['repo'], '--revision', manifest['revision'],
                            '--local-dir', str(directory)], check=True)
        count = verify(manifest, directory)
        print(f"Verified {role}: {count} files at {manifest['revision']}")


if __name__ == '__main__':
    main()
