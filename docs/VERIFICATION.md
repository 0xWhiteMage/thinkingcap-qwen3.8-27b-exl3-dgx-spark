# Recipe verification

## Tandem-first update: what was exercised

The primary recommendation is now [TandemLLM](TANDEM.md). Its new [identity manifest](../manifests/tandem.json) pins v0.12.0's engine/installer/community entry/target; it is not a separately frozen full auxiliary-asset manifest.

| Check | Result | Boundary |
| --- | --- | --- |
| Existing Spark source/profile/launcher/health inspection | Clean upstream v0.11.4 checkout; local rendered four-slot launcher, actual target symlink resolved | Not the stock v0.12.0 single-slot recipe |
| Actual resident target shards | Both sizes and SHA-256 values match the historical target manifest | Does not establish every tokenizer/base/drafter file's equivalence |
| Live raw and default-conductor requests, thinking off/on | Six final answers, zero errors/cap hits; raw injection zero, conductor packages present | Short diagnostics, not accuracy/throughput/concurrency qualification |
| Pinned v0.12.0 installer dry run on GB10 | Exit 0; candidate directory absent afterwards; existing engine retained | No real v0.12.0 installation, load, cold start or benchmark rerun |
| Niklas's PR #1 receipts | All 11 JSONs retained unchanged; SHA-256 values match his declarations | Independent contributor evidence, not a maintainer rerun |
| Probe raw-header regression | CLI transport test passes; no proxy needed for new raw requests | Test response is a fixture, not model evidence |

See [the sanitised inspection record](../evidence/tandem-inspection.json), [actual service analysis](SPARK-TANDEM.md) and [independent receipt index](receipts/README.md). Private reasoning is not included in the new inspection record. The existing service was not stopped, restarted, upgraded or switched for this documentation update. Recommending Tandem does not establish that it outperforms TensorFold on every workload.

## Historical TensorFold: what was exercised

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

The historical smoke requests used the then-resident qualified **TensorFold** runtime. **It was not stopped, replaced or restarted, and a second model was not loaded.** A replacement-image cold start and full benchmark rerun were not performed in that publication pass. These historical build/source-comparison results do not qualify Tandem. Reproduce the appropriate recipe on an idle Spark before a production deployment or performance comparison.

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
