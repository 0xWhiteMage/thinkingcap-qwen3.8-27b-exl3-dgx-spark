# TensorFold Python CUDA — legacy reproduction recipe

[README](../README.md) · [Recommended Tandem recipe](TANDEM.md) · [Methodology](BENCHMARKS.md)

**Retained for historical reproduction, not the primary serving recommendation.** TandemLLM is now the recommended Spark route. The existing Dockerfile, `scripts/serve.py`, target/drafter/runtime manifests and patches below continue to select **TensorFold**, not Tandem. They have not been silently repurposed.

The historical throughput, fidelity and reasoning-task measurements belong to this stack. They are not Tandem measurements. Do not substitute native TensorFold 1.x or a moving latest engine while keeping these claims; the Python CUDA base and credited patches are intentional pins.

## The legacy stack

```text
OpenAI-compatible client
        │
        ▼
TensorFold Python CUDA server
        │
        ├── ThinkingCap → Kearuga EXL3 target · nominal 4.00 decoder bpw
        ├── Spark: 16 lanes · 262,144-token window · stock DFlash 2
        ├── cuda24: experimental · 1 lane · ≤8,192 tokens · no drafter
        ├── BHCC2025 PR #260 EXL3 CUDA / prefill changes
        └── GB10 depth-planning admission · SM 12.1 enabled
```

| Component | Pinned input | What it contributes |
| --- | --- | --- |
| Target | [`0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw`](https://huggingface.co/0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw/tree/d6df32faa6a8109542b2d5e006d2a09312955905) | Published EXL3 weights, tokenizer, template and generation defaults |
| Target revision | `d6df32faa6a8109542b2d5e006d2a09312955905` | Immutable download and file-hash manifest |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | Stock DFlash 2; not the separately published Kearuga NVFP4 drafter |
| Drafter revision | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | Frozen stock mirror |
| TensorFold base | `ed78d6fc204d89d90b045bf033d6551e7714f3a1` | Upstream Python CUDA implementation; package version reports 0.6.6 |
| EXL3 changes | [BHCC2025's PR #260](https://github.com/ashhart/TensorFold/pull/260), vendored as a rebased patch | Prompt, multi-row verify and capacity changes; PR is closed, not merged |
| GB10 planning | [`0002-gb10-depth-planning.patch`](../patches/0002-gb10-depth-planning.patch) | Enables the existing depth planner for SM 12.1, alongside SM 12.0 |
| CUDA base | NVIDIA PyTorch `26.07-py3`, **pinned multi-architecture index** | Fixed linux/arm64 and linux/amd64 child manifests; digests in [runtime manifest](../manifests/runtime.json) |
| Spark serving profile | BF16 target KV, `--parallel 16`, `--context 262144` | Text-only qualification; model-default sampling is sent per request |
| Experimental discrete-GPU profile | BF16 KV, one lane, ≤8,192 context, no drafter | Guarded 24 GB attempt, not measured cold-start or inference support |

The Spark profile loads the pinned stock drafter with TensorFold's **4-bit drafter-linear setting** (`--drafter-bits 4`). The stock downloaded checkpoint remains unchanged. The experimental discrete-GPU profile does not load a drafter. Neither profile enables FP8 prompts, KV quantisation, vision or an alternative drafter.

## Requirements

- **Measured Spark profile:** one NVIDIA DGX Spark / GB10, Linux aarch64, 128 GB unified memory. No multi-Spark qualification is claimed.
- **Experimental 24 GB profile:** Linux x86-64 and an Ampere-or-newer discrete NVIDIA GPU. The image build is verified, but card fit and inference are not. See the [startup-admission limitation](../docs/24GB-GPUS.md#why-default-startup-is-blocked).
- Working Docker and NVIDIA Container Toolkit; the measured host used driver **580.173.02**. This is an observed version, not a certified minimum.
- Python 3.11+ on the host. The pinned container uses Python 3.12.
- Internet access for the NVIDIA base, upstream source and model snapshots. Hugging Face authentication is needed if repository access becomes restricted.
- At least **60 GB of free disk as a practical starting point**, plus headroom for container layers, compiler caches and future downloads. It is not an exact capacity guarantee.
- GPU capacity available: **do not start a second large model alongside an existing resident model.** Stop the other service intentionally first; this launcher will not do that for you.

The 16.35 GB figure is the two target weight shards, **not total runtime memory**. Drafter weights, KV, prompt states, workspaces and compilation buffers require additional memory. A 262,144-token per-request window does not imply sixteen fully populated maximum windows fit together.

## Quick start

This quick start is for the **Spark profile**. For a 24 GB discrete NVIDIA GPU, including RTX 3090/4090 or the Blackwell candidates RTX 5090 D V2 and RTX 5090 Laptop GPU, use the separate [24 GB guide](../docs/24GB-GPUS.md), not the 16-lane settings below. Run Spark commands **on the Spark**, not on a NAS or Windows build host.

### 1. Clone the recipe and install the download client

```bash
git clone https://github.com/0xWhiteMage/thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda.git
cd thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'huggingface-hub==1.24.0'
```

For long-term replay, record this recipe's own `git rev-parse HEAD` as well; the model and runtime pins alone do not freeze future recipe edits.

### 2. Download and verify both snapshots

```bash
python scripts/artifacts.py
```

This downloads the pinned target to `models/target` and the pinned drafter to `models/drafter`, then checks **all 33 manifest files**. LFS files use SHA-256; ordinary HF files use Git blob SHA-1 with its object header. Corruption or missing files stops the script. A later recheck needs no network:

```bash
python scripts/artifacts.py --verify-only
```

The target is also mirrored on [ModelScope AI](https://modelscope.ai/models/0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw) and [ModelScope CN](https://www.modelscope.cn/models/WhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw). Hugging Face is the canonical pinned source in the downloader; if using a mirror, place its target files in `models/target` and run the same verification. Never treat a moving mirror branch as the revision pin.

### 3. Build the pinned runtime

```bash
docker build --platform linux/arm64 -t thinkingcap-exl3:0.2.0 .
```

The build checks patch SHA-256 values, fetches the exact TensorFold commit, validates the rebased PR #260 source tree, applies the one-line GB10 planning change, pins runtime dependencies and checks the expected PyTorch/Triton versions. No model weights, auth tokens or local download caches enter the build context.

### 4. Inspect, then start

```bash
python scripts/serve.py --dry-run
python scripts/serve.py
```

The default is **localhost:8888**, with read-only model mounts and a persistent compilation cache. It reserves CPU cores `5-9,15-19`, leaving other Spark cores available; pass `--cpuset` if your allocation differs. The launcher checks architecture, GPU identity, image availability, model hashes, container-name collisions and the port before starting. It does **not** replace existing containers, stop another model, or alter SparkDash.

The first start can take several minutes while CUDA extensions compile and the runtime calibrates its planner. Readiness waits up to 45 minutes; a timeout retains the container for inspection rather than deleting evidence.

```bash
docker logs -f thinkingcap-exl3
curl -fsS http://127.0.0.1:8888/v1/models
```

For private-LAN access, `--host 0.0.0.0` is available, but **this recipe does not configure authentication**. Do not expose it to an untrusted network or the internet. Put authentication and an access-restricted proxy in front of it first. Docker uses host networking and host IPC to match the single-Spark profile; model storage remains read-only.

### 5. Run a real smoke request

```bash
python scripts/probe.py --thinking off --greedy --max-tokens 64 \
  --output results/smoke-ready.json
```

Check that the recorded final content is `READY`, finish reason is `stop`, and errors/cap hits are zero. A thinking-enabled four-request probe is:

```bash
python scripts/probe.py --thinking on --requests 4 --concurrency 4 \
  --max-tokens 1024 --output results/smoke-thinking-c4.json
```

These are small public diagnostics, **not a reproduction of the historical 50-prompt speed battery or a quality score**. Every receipt records TTFT, wall time, server-reported completion/reasoning tokens, final content, finish reason and cap hits. Choose a new output path for each run; receipts are never silently overwritten.

## API and lifecycle

The legacy endpoint defaults to `http://127.0.0.1:8888/v1`; the served model ID is `Qwen3.8 27B Kearuga (ThinkingCap)`. Do not send Tandem-specific headers. The default TensorFold launcher keeps 16 lanes and the 262,144-token main context; these settings do not transfer to Tandem.

```bash
docker stop thinkingcap-exl3
docker start thinkingcap-exl3
```

Before recreating with changed options, deliberately remove only the stopped recipe container and rerun the launcher. Do not prune unrelated containers/caches. Never start this on the same Spark as a resident Tandem model.

For an experimental 24 GB discrete GPU, follow the [separate guide](24GB-GPUS.md). Niklas's measured RTX 5090 is the **32 GB desktop card using ExLlamaV3 + MTP**, not this guarded TensorFold profile; see [independent recipes](RECOMMENDED-RECIPES.md).

[Verification](VERIFICATION.md), [provenance](PROVENANCE.md), [troubleshooting](TROUBLESHOOTING.md) and [third-party notices](../THIRD_PARTY_NOTICES.md) retain the original evidence and licence boundaries.
