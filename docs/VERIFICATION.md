# Recipe verification

## What was exercised

| Check | Result | Boundary |
| --- | --- | --- |
| Pinned target download, sizes and hashes | All 28 manifest files verified | Published snapshot, not re-quantisation |
| Pinned drafter download, sizes and hashes | All 5 manifest files verified | Stock checkpoint, not a custom drafter |
| ARM64 Docker build on a real DGX Spark | Passed | CUDA base, exact source fetch, patch hashes/tree, install, Python compilation and serving CLI |
| AMD64 Docker build | Passed | Same pinned CUDA index/source; image builds and CLI works, not GPU execution |
| EXL3 linear extension for SM 8.6 / 8.9 | Compiled, linked and CPU-loaded; both architecture cubins inspected | Not RTX kernel execution, whole-runtime CUDA qualification or measured fit |
| Live preflight without an exposed GPU | Refuses before reading/loading model weights | Missing-device guard, not a real-card free-memory test |
| Exact-runtime 24 GiB header-capacity calculation | Default 20 GiB budget refuses all tested contexts; ideal 22 GiB override admits 8,192 | Header/source estimate only; no GPU exposed and no weights allocated |
| Installed source/data comparison with the qualified resident runtime | All 456 files match after CRLF-to-LF normalisation | Not raw-byte or Docker image-ID equality |
| Live thinking-off smoke | Final content `READY`, zero errors and zero cap hits | One short API request, not a speed/quality qualification |
| Live thinking-on C4 smoke | Four final answers, zero errors and zero cap hits | New public diagnostic prompts, not the historical battery |
| Local regression tests and byte compilation | Passed | Test fixtures are unit-test data, not model results |
| Original ThinkingCap licence bundle | Exact pinned bytes retained and regression-tested | Covers the reused release materials as well as model notices |

The smoke requests used the existing resident qualified runtime. **It was not stopped, replaced or restarted, and a second model was not loaded.** A replacement-image cold start and full benchmark rerun were not performed in this publication pass. Reproduce the recipe on an idle Spark before using it for a production deployment or comparing performance claims.

## Evidence files

- [Build and verification receipt](../evidence/verification.json)
- [Thinking-off stream receipt](../evidence/smoke-ready.json)
- [Thinking-on C4 stream receipt](../evidence/smoke-thinking-c4.json)
- [Historical source measurements](../evidence/source/BENCHMARKS.json)
- [Historical release provenance](../evidence/source/RELEASE_PROVENANCE.json)
- [24 GiB header/source estimates](../evidence/24gb-header-estimates.json)
- [x86 image, compiled architecture cubins and missing-GPU guard](../evidence/x86-build-and-kernels.json)

Live receipts contain server-reported usage, actual reasoning/final output, finish reasons and observed timings. Their aggregate speed is deliberately **not promoted to a headline benchmark**; the READY request is only two completion tokens and the four prompts have short, unequal replies.

## Re-run local checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/serve.py --dry-run
python3 scripts/serve.py --profile cuda24 --dry-run
python3 scripts/artifacts.py --verify-only
```

The GitHub workflow checks scripts/tests, manifest identities, patch checksums and local documentation links. It does not claim GPU inference or re-download 20 GB of weights on hosted CI.

Optional comparison, if you already have the qualified runtime running:

```bash
python3 scripts/compare_runtime.py --container YOUR_CONTAINER --normalise-line-endings
```

That comparison starts only an ephemeral **CPU-side image-inspection container**, without `--gpus` or model mounts. It compares source/data checksums after normalising CRLF; generated Python bytecode and compiler caches are excluded. Omit the normalisation flag to require raw-byte equality. Do not use this optional comparison as a prerequisite for a fresh installation.
