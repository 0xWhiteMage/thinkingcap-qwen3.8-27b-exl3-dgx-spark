import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArtifactTests(unittest.TestCase):
    def test_git_blob_header_is_verified(self):
        module = load_script('artifacts')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'LICENSE').write_bytes(b'abc')
            manifest = {'files': [{'path': 'LICENSE', 'size': 3,
                                  'git_blob_sha1': hashlib.sha1(b'blob 3\0abc').hexdigest()}]}
            self.assertEqual(module.verify(manifest, root), 1)

    def test_verification_rejects_corrupted_file(self):
        module = load_script('artifacts')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_bytes(b'original')
            manifest = {'files': [{'path': 'config.json', 'size': 8,
                                  'sha256': hashlib.sha256(b'original').hexdigest()}]}
            module.verify(manifest, root)
            (root / 'config.json').write_bytes(b'corrupt!')
            with self.assertRaises(ValueError):
                module.verify(manifest, root)


class RepositoryTests(unittest.TestCase):
    def test_redistributed_release_keeps_exact_licence_bundle(self):
        module = load_script('artifacts')
        names = {'LICENSE', 'LICENSE-PolyForm-Small-Business-1.0.0.txt',
                 'LICENSE-Apache-2.0-Qwen.txt', 'NOTICE', 'ATTRIBUTIONS.md'}
        manifest = json.loads((ROOT / 'manifests' / 'target.json').read_text())
        subset = {'files': [entry for entry in manifest['files'] if entry['path'] in names]}
        self.assertEqual(len(subset['files']), len(names))
        self.assertEqual(module.verify(subset, ROOT / 'third_party' / 'thinkingcap'), len(names))

    def test_manifests_are_pinned_and_complete(self):
        import re
        count = 0
        for role in ['target', 'drafter']:
            manifest = json.loads((ROOT / 'manifests' / (role + '.json')).read_text())
            self.assertRegex(manifest['revision'], r'^[0-9a-f]{40}$')
            paths = [entry['path'] for entry in manifest['files']]
            self.assertEqual(len(paths), len(set(paths)))
            count += len(paths)
            for entry in manifest['files']:
                self.assertGreater(entry['size'], 0)
                self.assertFalse(Path(entry['path']).is_absolute())
                self.assertNotIn('..', Path(entry['path']).parts)
        self.assertEqual(count, 33)

    def test_patch_hashes(self):
        for line in (ROOT / 'patches' / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split()
            self.assertEqual(hashlib.sha256((ROOT / 'patches' / name).read_bytes()).hexdigest(), digest)

    def test_local_documentation_targets_exist(self):
        import re
        pages = [ROOT / 'README.md', ROOT / 'THIRD_PARTY_NOTICES.md', *sorted((ROOT / 'docs').glob('*.md'))]
        for page in pages:
            for target in re.findall(r'\]\(([^)\s]+)\)', page.read_text()):
                if '://' in target or target.startswith('#'):
                    continue
                self.assertTrue((page.parent / target.split('#')[0]).exists(), f'{page}: {target}')


class LauncherTests(unittest.TestCase):
    def test_dry_command_keeps_weights_read_only_and_local(self):
        import argparse
        module = load_script('serve')
        args = argparse.Namespace(models_dir=Path('/test/models'), cache_dir=Path('/test/cache'),
                                  image='recipe:test', name='recipe-test', host='127.0.0.1',
                                  port=18888, parallel=16, context=262144, cpuset='5-9,15-19')
        command = module.command(args)
        self.assertIn('/test/models/target:/models/target:ro', command)
        self.assertIn('/test/models/drafter:/models/drafter:ro', command)
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
        self.assertEqual(command[command.index('--context') + 1], '262144')
        self.assertNotIn('--privileged', command)


class ProbeTests(unittest.TestCase):
    def events(self, *objects):
        for item in objects:
            yield ('data: ' + json.dumps(item) + '\n\n').encode()
        yield b'data: [DONE]\n\n'

    def test_preserves_reasoning_and_token_usage(self):
        module = load_script('probe')
        result = module.parse_stream(self.events(
            {'choices': [{'delta': {'reasoning_content': 'think'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': 'READY'}, 'finish_reason': 'stop'}]},
            {'choices': [], 'usage': {'prompt_tokens': 3, 'completion_tokens': 9,
                                    'completion_tokens_details': {'reasoning_tokens': 4}}}))
        self.assertEqual(result['content'], 'READY')
        self.assertEqual(result['reasoning_content'], 'think')
        self.assertEqual(result['usage']['completion_tokens'], 9)

    def test_missing_usage_is_not_zero_tokens(self):
        module = load_script('probe')
        with self.assertRaises(ValueError):
            module.parse_stream(self.events({'choices': [{'delta': {'content': 'x'}, 'finish_reason': 'stop'}]}))

    def test_server_error_is_not_success(self):
        module = load_script('probe')
        with self.assertRaises(ValueError):
            module.parse_stream(self.events({'error': {'message': 'no capacity'}}))

    def test_missing_done_is_incomplete(self):
        module = load_script('probe')
        with self.assertRaises(ValueError):
            module.parse_stream([])


if __name__ == '__main__':
    unittest.main()
