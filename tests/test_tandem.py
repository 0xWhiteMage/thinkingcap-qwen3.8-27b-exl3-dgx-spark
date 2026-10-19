"""Tandem transport regression tests. Responses below are fixtures, not model evidence."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TandemProbeTests(unittest.TestCase):
    def test_raw_cli_sends_header_and_records_transport(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.path, self.headers.get('X-Tandem'), payload))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                event = {'choices': [{'delta': {'content': 'READY'}, 'finish_reason': 'stop'}],
                         'usage': {'completion_tokens': 2, 'prompt_tokens': 5}}
                self.wfile.write(('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode())

            def log_message(self, format, *args):
                pass

        with HTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / 'fixture.json'
                    result = subprocess.run(
                        [sys.executable, str(ROOT / 'scripts' / 'probe.py'), '--tandem-raw',
                         '--endpoint', f'http://127.0.0.1:{server.server_port}', '--model', 'TandemLLM',
                         '--thinking', 'off', '--greedy', '--max-tokens', '64', '--output', str(output)],
                        capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(output.read_text())
                    self.assertTrue(report['settings']['tandem_raw'])
                    self.assertEqual(report['summary']['errors'], 0)
            finally:
                server.shutdown()
                thread.join(timeout=5)
        self.assertEqual(len(received), 1)
        path, header, payload = received[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(header, 'raw')
        self.assertEqual(payload['model'], 'TandemLLM')
        self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])


class TandemEvidenceTests(unittest.TestCase):
    def test_independent_receipts_are_complete_and_unchanged(self):
        root = ROOT / 'docs' / 'receipts'
        entries = [line.split() for line in (root / 'SHA256SUMS').read_text().splitlines()]
        self.assertEqual(len(entries), 11)
        expected = {name for digest, name in entries}
        actual = {path.relative_to(root).as_posix() for path in root.glob('*/*.json')}
        self.assertEqual(actual, expected)
        report = (root / 'niklas-pr-1-report.md').read_text()
        for digest, name in entries:
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
            self.assertIn(digest, report)

    def test_tandem_identity_is_pinned_separately_from_resident_service(self):
        manifest = json.loads((ROOT / 'manifests' / 'tandem.json').read_text())
        for section in ['engine', 'target']:
            self.assertRegex(manifest[section]['revision'], r'^[0-9a-f]{40}$')
        for section in ['installer', 'profile']:
            self.assertRegex(manifest[section]['sha256'], r'^[0-9a-f]{64}$')
        self.assertEqual(manifest['engine']['version'], 'v0.12.0')
        self.assertEqual(manifest['verification']['resident_smoke_version'], 'v0.11.4')
        self.assertFalse(manifest['verification']['fresh_install_and_cold_start_repeated_here'])

    def test_sanitised_live_evidence_covers_both_paths_and_thinking_modes(self):
        evidence = json.loads((ROOT / 'evidence' / 'tandem-inspection.json').read_text())
        runs = evidence['smoke_runs']
        self.assertEqual(sum(run['requests'] for run in runs), 6)
        self.assertEqual({(run['path'], run['settings']['thinking']) for run in runs},
                         {('raw', 'on'), ('raw', 'off'), ('default-conductor', 'on'),
                          ('default-conductor', 'off')})
        for run in runs:
            self.assertEqual(run['errors'], 0)
            self.assertEqual(run['cap_hits'], 0)
            self.assertEqual(run['requests'], len(run['records']))
            for record in run['records']:
                self.assertEqual(record['finish_reason'], 'stop')
                self.assertTrue(record['has_final_answer'])
                self.assertNotIn('reasoning_content', record)
                self.assertNotIn('content', record)
                if run['path'] == 'raw':
                    self.assertEqual(record['injected_tokens'], 0)
                    self.assertFalse(record['harness_present'])


if __name__ == '__main__':
    unittest.main()
