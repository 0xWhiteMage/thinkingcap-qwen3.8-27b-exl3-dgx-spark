<div align="center">

![Kearuga — ThinkingCap Qwen3.8 27B](assets/header.png)

# ThinkingCap EXL3 · DGX Spark & 24 GB NVIDIA GPUs

**Compact weights. Efficient reasoning. Pinned TensorFold CUDA for Spark and experimental 24 GB GPU qualification.**

[![Hardware](https://img.shields.io/badge/DGX_Spark-GB10-76b900)](#requirements)
[![24 GB GPUs](https://img.shields.io/badge/24_GB_NVIDIA-experimental-d69e2e)](#24-gb-nvidia-gpus)
[![Format](https://img.shields.io/badge/EXL3-4.00_bpw-7854d8)](#the-stack)
[![Runtime](https://img.shields.io/badge/TensorFold-Python_CUDA-275cbd)](#the-stack)
[![Weights](https://img.shields.io/badge/Hugging_Face-weights-ffd21e)](https://huggingface.co/0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw)
[![Tests](https://github.com/0xWhiteMage/thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda/actions/workflows/checks.yml/badge.svg)](https://github.com/0xWhiteMage/thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda/actions/workflows/checks.yml)
[![Follow on X](https://img.shields.io/badge/X-@0xWhiteMage-000000?logo=x&logoColor=white)](https://x.com/0xWhiteMage)
[![Donate on Ko-fi](https://img.shields.io/badge/Ko--fi-Donate-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/0xwhitemage)

[Why this recipe](#why-this-recipe) · [The stack](#the-stack) · [Spark results](#measured-results) · [Spark quick start](#quick-start) · [24 GB GPUs](#24-gb-nvidia-gpus) · [API](#use-the-api) · [Reproducibility](#reproducibility-boundaries) · [Credits](#credits-and-licences)

</div>

---

Serve **BottleCap AI's ThinkingCap-Qwen3.8-27B**, quantised as the **Kearuga EXL3 4.00bpw release**, through TensorFold's local OpenAI-compatible CUDA endpoint. This recipe provides the measured **DGX Spark profile** and a separate, guarded **24 GB NVIDIA qualification path**. It pins model inputs, rebuilds the Python CUDA runtime from public source plus credited patches, and checks the bytes before loading them.

> **24 GB is experimental, not qualified support.** The x86-64 image builds, but the pinned runtime's default loading-admission estimate exceeds a 24 GiB card's default budget. Read the [24 GB guide](docs/24GB-GPUS.md) before attempting it. All throughput figures below are **Spark-only**.

| Your hardware | Start here | Current status |
| --- | --- | --- |
| DGX Spark / GB10, 128 GB unified memory | [Spark quick start](#quick-start) | Measured serving profile; historical Spark benchmarks retained |
| 24 GB discrete NVIDIA GPUs: RTX 3090 / 4090, RTX 5090 D V2 and RTX 5090 Laptop GPU | [24 GB NVIDIA GPU guide](docs/24GB-GPUS.md) | Experimental one-GPU path; Ampere/Ada EXL3 compilation verified. Blackwell candidates are not yet qualified; SM 12.0 compilation, cold start and fit are unverified |

| Weight storage | Mean KL vs ThinkingCap BF16 | Single-request net decode | 16-request aggregate throughput |
| :---: | :---: | :---: | :---: |
| **16.35 GB** | **0.00517** | **60.61 tok/s** | **249.55 tok/s** |

> **Read these as historical measurements, not installation promises.** The figures are from the release's 10–11 October 2026 qualification. C1 net decode excludes TTFT; C16 is an aggregate greedy descriptive screen, not one person's decode speed. The internal fidelity and task suites are not bundled. See [methodology](docs/BENCHMARKS.md) and the unchanged [source benchmark record](evidence/source/BENCHMARKS.json).

## Why this recipe

- **Keep the reference close.** The release measured 97.7863% top-1 token agreement with ThinkingCap BF16 at teacher-forced positions. That is fidelity, not task accuracy or identical free-running answers.
- **Use one Spark for shared serving.** The recorded stack improved C16 aggregate throughput from 181.91 to 249.55 tok/s using the same target and stock drafter. The gain belongs to the full serving stack, not quantisation alone.
- **Avoid moving dependencies.** EXL3 support here lives in TensorFold's Python CUDA branch. Installing native TensorFold 1.x, SGLang or vLLM is not an equivalent route to this checkpoint.
- **Build from auditable inputs.** The Dockerfile validates the patched source tree; the downloader validates every pinned snapshot file; the API probe refuses to turn missing token usage or server errors into a speed result.
- **Inspect the 24 GB boundary honestly.** A one-lane, no-drafter profile checks real free memory before launch; header estimates and architecture builds are not presented as measured RTX support.

This is a **serving recipe for the published EXL3 weights**, not a new ThinkingCap fine-tune or a BF16-to-EXL3 conversion/calibration recipe.

## The stack

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
| GB10 planning | [`0002-gb10-depth-planning.patch`](patches/0002-gb10-depth-planning.patch) | Enables the existing depth planner for SM 12.1, alongside SM 12.0 |
| CUDA base | NVIDIA PyTorch `26.07-py3`, **pinned multi-architecture index** | Fixed linux/arm64 and linux/amd64 child manifests; digests in [runtime manifest](manifests/runtime.json) |
| Spark serving profile | BF16 target KV, `--parallel 16`, `--context 262144` | Text-only qualification; model-default sampling is sent per request |
| Experimental discrete-GPU profile | BF16 KV, one lane, ≤8,192 context, no drafter | Guarded 24 GB attempt, not measured cold-start or inference support |

The Spark profile loads the pinned stock drafter with TensorFold's **4-bit drafter-linear setting** (`--drafter-bits 4`). The stock downloaded checkpoint remains unchanged. The experimental discrete-GPU profile does not load a drafter. Neither profile enables FP8 prompts, KV quantisation, vision or an alternative drafter.

## Measured results

### Serving throughput

| Workload | Previous TensorFold stack | Qualified stack | Metric |
| --- | ---: | ---: | --- |
| 1 request | 59.43 tok/s | **60.61 tok/s** | Net decode, TTFT excluded |
| 4 concurrent requests | 124.76 tok/s | **137.59 tok/s** | Aggregate completion tokens / cell wall time |
| 8 concurrent requests | 167.98 tok/s | **214.88 tok/s** | Aggregate, greedy descriptive |
| 16 concurrent requests | 181.91 tok/s | **249.55 tok/s** | Aggregate, greedy descriptive |
| Cold long-prompt median TTFT | 10.535 s | **7.448 s** | First-token wait; not sustained decode |

![Serving performance — historical release measurements](assets/serving-performance.png)

### Fidelity and task timing

| Fidelity measure | EXL3 result | Scope |
| --- | ---: | --- |
| Mean coarsened KL | 0.00517361 | Fixed BF16 top-20 support + other bucket |
| Pooled position p99 KL | 0.06194665 | Tail divergence |
| Top-1 token agreement | 97.7863% | Teacher-forced positions, not task accuracy |
| Tool-domain mean KL | 0.00538761 | Tool slice, not full tool-use certification |
| Maximum position KL | 1.88004674 | Individual positions can differ substantially |

The internal fidelity suite covers **240 prompts and 25,780 positions**. Its reference is ThinkingCap BF16, not base Qwen BF16. KL numbers from other cards are not comparable without matching the reference, support, prompts and method.

| Reasoning workload | Previous mean attempt time | Qualified mean attempt time |
| --- | ---: | ---: |
| MATH-500 Level-5 fixed 40-problem sample | 46.10 s | **40.17 s** |
| AIME 2025, 30 problems | 301.74 s | **291.83 s** |
| Combined 70-problem suite | 155.66 s | **148.02 s** |

Three repeats averaged **62/70 correct**, versus 60/70 previously. The registered quality verdict was **non-inferior**, not general superiority. Wrong and capped attempts remain in the timing averages; current AIME runs reached the cap on 5–6 problems per run. These are measured attempt times, not token-rate estimates or successful-task-only latency.

![Reasoning attempt latency — historical release measurements](assets/task-latency.png)

### ThinkingCap's contribution

BottleCap reports **37.2% fewer reasoning tokens** on a macro-average BF16 evaluation, with 85.8% average accuracy versus base Qwen's 86.6%. Those are [upstream publisher results](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B/blob/7891e03dbda2e9b63a5b9610f20e83ea441f8e2c/README.md), not EXL3 measurements; the upstream card may require approved Hugging Face access. This recipe claims no additional thinking-token reduction: the internal 70-problem suite averaged 5,132 reasoning tokens per attempt versus 5,095 previously. Do not multiply an upstream token reduction into the serving-speed improvement.

## Requirements

- **Measured Spark profile:** one NVIDIA DGX Spark / GB10, Linux aarch64, 128 GB unified memory. No multi-Spark qualification is claimed.
- **Experimental 24 GB profile:** Linux x86-64 and an Ampere-or-newer discrete NVIDIA GPU. The image build is verified, but card fit and inference are not. See the [startup-admission limitation](docs/24GB-GPUS.md#why-default-startup-is-blocked).
- Working Docker and NVIDIA Container Toolkit; the measured host used driver **580.173.02**. This is an observed version, not a certified minimum.
- Python 3.11+ on the host. The pinned container uses Python 3.12.
- Internet access for the NVIDIA base, upstream source and model snapshots. Hugging Face authentication is needed if repository access becomes restricted.
- At least **60 GB of free disk as a practical starting point**, plus headroom for container layers, compiler caches and future downloads. It is not an exact capacity guarantee.
- GPU capacity available: **do not start a second large model alongside an existing resident model.** Stop the other service intentionally first; this launcher will not do that for you.

The 16.35 GB figure is the two target weight shards, **not total runtime memory**. Drafter weights, KV, prompt states, workspaces and compilation buffers require additional memory. A 262,144-token per-request window does not imply sixteen fully populated maximum windows fit together.

## Quick start

This quick start is for the **Spark profile**. For a 24 GB discrete NVIDIA GPU, including RTX 3090/4090 or the Blackwell candidates RTX 5090 D V2 and RTX 5090 Laptop GPU, use the separate [24 GB guide](docs/24GB-GPUS.md), not the 16-lane settings below. Run Spark commands **on the Spark**, not on a NAS or Windows build host.

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

## 24 GB NVIDIA GPUs

There is a separate x86-64 build and **opt-in `cuda24` profile** for users evaluating 24 GB discrete NVIDIA GPUs. Examples include RTX 3090/4090 and the **RTX 5090 D V2 / RTX 5090 Laptop GPU — 24 GB Blackwell candidates, not yet qualified**. The profile selects one GPU, removes the drafter, uses one lane with at most 8,192 context tokens, keeps BF16 KV/prompt math, and checks the upstream memory plan before starting. Its architecture floor does not exclude Blackwell, but SM 12.0 compilation and real-card startup/inference have not been tested; Spark's SM 12.1 results do not establish RTX 50-series support.

**The limitation is startup, not just context.** Header-derived estimates give 17.085 GiB while serving at 8,192 context, but **21.898 GiB at loading**. TensorFold's default reserve leaves at most 20 GiB on an ideal 24 GiB card, so default admission refuses even a short context. The upstream 2 GiB reserve override needs at least 23.898 GiB truly free after CUDA initialisation—too little spare room to promise a normal desktop card will fit.

```bash
# Target only; do not download or mount a speculative drafter for this profile.
python scripts/artifacts.py --role target
docker build --platform linux/amd64 -t thinkingcap-exl3:0.2.0 .
python scripts/serve.py --profile cuda24 --dry-run
```

The actual attempt requires `--experimental-24gb` and may safely refuse. See the [full 24 GB guide](docs/24GB-GPUS.md) for architecture differences, live preflight, requests without prefix retention, and the evidence needed to qualify a card. **No RTX speed or measured-fit claim is made.**

## Use the API

This is the **Spark** API example. The 24 GB guide uses a smaller output default/request budget, serial requests and `"draft": false`; the output default is not a hard per-request ceiling.

```bash
curl -N http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8 27B Kearuga (ThinkingCap)",
    "messages": [{"role": "user", "content": "Explain Python context managers with a short example."}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}
  }'
```

Reasoning is returned in `reasoning_content`, separately from final `content`. `max_tokens` includes both: an undersized budget can end before the answer. `xhigh` follows the upstream recommendation; evaluate `medium` or `low` on your workload rather than assuming equal quality. Client settings can override server defaults, so send the intended sampling and thinking fields explicitly.

## Reproducibility boundaries

| What this repository provides | What it does not establish |
| --- | --- |
| Immutable target/drafter manifests and checked downloads | A BF16 conversion/calibration dataset or exact re-quantisation procedure |
| Buildable public-source TensorFold Python CUDA stack | Support in native TensorFold 1.x, SGLang or vLLM |
| PR #260 attribution, patch checksums and source-tree check | Kearuga authorship of community kernels |
| Pinned ARM64/AMD64 CUDA base manifests and runtime dependency versions | Byte-identical Docker image IDs or identical timing on all hosts |
| Safe launcher and public API probes | Reproduction of unpublished fidelity prompts/captures or internal Hard-V2-R sampling |
| Fresh ARM64 image build, source comparison and live API receipts | A fresh replacement-server cold start or a rerun of the full benchmark qualification |
| Fresh x86-64 image build and reproducible header-capacity estimates | A measured 24 GB cold start, peak memory, inference pass or RTX benchmark |

The recipe image was built on the Spark. Its **456 installed source/data files match the resident qualified runtime after CRLF-to-LF normalisation**; this is source-text equivalence, not raw-byte equality. The live smoke probes used that existing resident runtime so a healthy service did not need to be restarted. See [verification evidence](docs/VERIFICATION.md). No new performance or general-quality claim is made from those smoke requests.

## Repository map

```text
Dockerfile                 Pinned ARM64/AMD64 build from public TensorFold source
requirements-runtime.txt   Locked Python runtime dependency closure
manifests/                 Immutable target, drafter and runtime identities
patches/                   Credited PR #260 rebase, GB10 change, checksums
scripts/artifacts.py       Download and verify model files
scripts/serve.py           Spark launcher and guarded experimental discrete-GPU profile
scripts/capacity_check.py  Exact-runtime header estimates and live admission preflight
scripts/probe.py           Streaming API probe and honest timing receipts
scripts/compare_runtime.py Optional source-text comparison with an existing runtime
bench/prompts.json         Small public prompt set, not the historical benchmark suite
assets/                    Release banner and historical charts
evidence/                  Live smoke receipts and unchanged source benchmark records
docs/                      Methodology, provenance, verification, troubleshooting
third_party/tensorfold/    Retained upstream software licence and notices
third_party/thinkingcap/   Unchanged target-release licence bundle and credits
```

## Troubleshooting

Start with [the troubleshooting guide](docs/TROUBLESHOOTING.md): wrong image architecture, EXL3/runtime mismatch, failed hashes, first-boot compilation, missing reasoning answers, and port/container collisions all have separate remedies. **Do not “fix” a failure by loading different weights or a latest engine while retaining the old benchmark claims.**

To stop this recipe's own container:

```bash
docker stop thinkingcap-exl3
```

To resume the same container, use `docker start thinkingcap-exl3`. To recreate it with changed options, explicitly remove the stopped container with `docker rm thinkingcap-exl3`, then rerun the launcher. Model downloads and compiler caches stay outside the container. Never use a broad prune command as a startup step.

## Credits and licences

| Creator / contributor | Contribution |
| --- | --- |
| **BottleCap AI** | ThinkingCap fine-tune, source weights, token-efficiency work and governing model licence |
| **Alibaba Cloud / Qwen Team** | Original Qwen model, architecture, tokenizer and Apache-2.0 upstream materials |
| **Turboderp / ExLlamaV3 contributors** | EXL3 format and quantisation tooling |
| **Ash (`ashhart`) / TensorFold contributors** | Serving engine, scheduling, verification and sampling machinery |
| **BHCC2025 / BHCC Spark Lab** | PR #260 EXL3 CUDA contributions used by the qualified runtime |
| **Inco AI; z-lab / Z Lab** | DFlash 2, stock mirror and original DFlash research/architecture |
| **khronnuz** | Community model resources used during preparation/evaluation, not the final direct source weights |
| **0xWhiteMage / Kearuga** | Quantised release, evaluation, serving integration and this recipe |
| **NVIDIA, PyTorch, Triton contributors** | Hardware, CUDA environment and software toolchain |

Required Notice: Copyright 2026 BottleCap AI (https://bottlecapai.com)

Authored recipe scripts and documentation use [MIT](LICENSE). Vendored TensorFold changes remain **Apache-2.0**, with [upstream licence and notices retained](third_party/tensorfold/). The weights and reused release materials retain **PolyForm Small Business 1.0.0 plus BottleCap's individual personal, non-commercial-use permission**; the [original licence and permission](third_party/thinkingcap/LICENSE), [full PolyForm terms](third_party/thinkingcap/LICENSE-PolyForm-Small-Business-1.0.0.txt), [Qwen licence](third_party/thinkingcap/LICENSE-Apache-2.0-Qwen.txt), [NOTICE](third_party/thinkingcap/NOTICE) and [attributions](third_party/thinkingcap/ATTRIBUTIONS.md) are included unchanged. Qwen's Apache-2.0 materials do not make the whole derivative Apache-2.0. Commercial use outside the Small Business definition requires separate permission from BottleCap. External drafter/software licences are separate; no weights are redistributed in this Git repository. See [third-party scope](THIRD_PARTY_NOTICES.md).

No contributor endorsement is implied. Please credit ThinkingCap, Qwen, EXL3, TensorFold and DFlash when discussing their work.
