# Recommended recipes and independent receipts

[README](../README.md) · [Primary Tandem recipe](TANDEM.md) · [Receipt index and checksums](receipts/README.md)

**Benchmarks by Niklas Lenz** — [GitHub: Nipale-ai](https://github.com/Nipale-ai) · [X: @niklaslenz_ai](https://x.com/niklaslenz_ai) · [niklaslenz.com](https://niklaslenz.com). Thank you to Niklas for independently testing the release, supplying receipts and documenting the limits as well as the results.

These are **suggested hardware-specific routes**, not interchangeable certified profiles. The 11 original JSON receipts from [PR #1](https://github.com/0xWhiteMage/thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda/pull/1) are preserved unchanged at head `031ebfda6b840f66dd6728f4b69ced5e5ea9dcc2`; all match the author's declared SHA-256 values. The [complete contributor report](receipts/niklas-pr-1-report.md) retains the rest of his setup, methods, proposed wording, proxy source and caveats. Runs are dated 19 October 2026 by the author.

The JSONs establish the recorded client/API outputs, usage, timings and finish states. Hardware, driver, clock, engine commit, weight verification, idle-state and log-based claims remain **author-reported provenance**; they are not independently established by the attached probe files. Our separately inspected live Spark has its own labelled evidence.

## Choose the right route

| Hardware / purpose | Recipe location | Evidence boundary |
| --- | --- | --- |
| DGX Spark / ASUS GX10, GB10, 128 GB unified memory | **[TandemLLM primary recipe](TANDEM.md)** and [upstream community profile](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/models/qwen3.8-27b/community/thinkingcap-27b-kearuga-exl3-4.00.json) | Independent v0.12.0 C1 raw-path receipt; our resident v0.11.4 inspection is a separate configuration |
| Desktop **RTX 5090 32 GB** | [Niklas's setup below](#rtx-5090-32-gb--exllamav3--mtp) and [MiaAI-Lab kit at the measured revision](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/tree/bbb11db4b7ff688a62841c68e0892fa099bc3362) | ExLlamaV3 1.6.0 + MTP, batch one, int4 KV; **not Tandem or TensorFold** |
| Historical Spark reproduction | [Legacy TensorFold recipe](TENSORFOLD.md) | Independent public C1/C4/C8/C16 cells plus historical release evidence; different prompt suites |
| RTX 3090/4090, RTX 5090 D V2 or RTX 5090 Laptop **24 GB** | [Experimental 24 GB guide](24GB-GPUS.md) | Existing TensorFold admission path, not qualified by a 32 GB desktop 5090 receipt |

No Tandem-on-RTX-5090 result is supplied. Spark's SM 12.1 execution does not qualify consumer SM 12.0. A configured 262,144-token context is not evidence of a full-window quality or memory stress test.

## Receipt-backed public probe results

Each performance cell below has 16 requests, thinking on, a 1,024-token cap and zero request errors. Sampling is temperature 1.0 / top-p 0.95 / top-k 20; greedy uses temperature 0. The probe sends `enable_thinking=true` and nested `reasoning_effort=xhigh`; that requested value is not independent verification of each stack's effective effort.

| Hardware / engine | Cell | Aggregate tok/s | Derived C1 net decode tok/s | Mean TTFT | Cap hits | Receipt |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RTX 5090, ExLlamaV3 + MTP | C1 sampled | 151.10 | 161.74 | 0.092 s | 1 | [JSON](receipts/rtx5090/c1-sampled.json) |
| RTX 5090, ExLlamaV3 + MTP | C1 greedy | 164.08 | 178.92 | 0.088 s | 0 | [JSON](receipts/rtx5090/c1-greedy.json) |
| GB10, Tandem v0.12.0 **raw** | C1 sampled | 62.22 | 66.02 | 0.158 s | 0 | [JSON](receipts/dgx-spark-tandemllm/sampled-c1-tandem.json) |
| GB10, TensorFold Python CUDA | C1 sampled | 52.43 | 60.16 | 0.506 s | 0 | [JSON](receipts/dgx-spark-tensorfold/sampled-c1.json) |
| GB10, TensorFold Python CUDA | C1 greedy | 57.49 | 69.62 | 0.511 s | 0 | [JSON](receipts/dgx-spark-tensorfold/public-c1.json) |
| GB10, TensorFold Python CUDA | C4 greedy | 129.51 | — | 0.627 s | 0 | [JSON](receipts/dgx-spark-tensorfold/public-c4.json) |
| GB10, TensorFold Python CUDA | C8 greedy | 166.33 | — | 0.950 s | 0 | [JSON](receipts/dgx-spark-tensorfold/public-c8.json) |
| GB10, TensorFold Python CUDA | C16 greedy | 212.10 | — | 1.903 s | 0 | [JSON](receipts/dgx-spark-tensorfold/public-c16.json) |

**Do not rank the engines from this table.** Hardware, KV, drafter, clocks, caches, generated text and serving paths differ. Niklas's Spark was capped at **2200 MHz**; the desktop 5090 had no clock cap. Each cell is one boot / one run, not a multi-boot confidence interval. The sampled Tandem and TensorFold cells generated 2,726 and 3,306 completion tokens respectively.

- Aggregate = all completion tokens / total probe wall time, including request overhead.
- Derived C1 net decode = sum(completion tokens) / sum(request wall − client TTFT). Recalculated from the JSON; the original probe does not produce this metric. The same division at C4+ is **not** a clean per-user decode rate, so it is omitted here.
- TTFT is client-observed first non-empty reasoning or answer text, not an engine-only prefill timer.
- Reasoning contributes to completion tokens. The 5090 kit does not report its reasoning-token share. Its sampled `code-race` request reached the cap without a final answer; those 1,024 tokens remain in the speed aggregate.
- The [readiness](receipts/rtx5090/smoke-ready.json), [TensorFold readiness](receipts/dgx-spark-tensorfold/smoke-ready.json) and [TensorFold C4 smoke](receipts/dgx-spark-tensorfold/smoke-thinking-c4.json) receipts are diagnostics, not headline benchmarks.

## RTX 5090 32 GB — ExLlamaV3 + MTP

**Suggested independent recipe, not a maintainer-requalified install.** Niklas used ExLlamaV3 **1.6.0** through the [MiaAI-Lab one-click kit](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/tree/bbb11db4b7ff688a62841c68e0892fa099bc3362) at `bbb11db4b7ff688a62841c68e0892fa099bc3362`, driver `610.43.02`, Linux `6.12.95+deb13-amd64`. Target HF revision: `11447639033dd80a4c04536b5b37ed7d642802ee`, with 26/26 files verified against that snapshot's `SHA256SUMS` according to his report. This does not mean the recipe's older 28-file manifest was used.

Inspect the kit's pinned README/install scripts and licence/dependencies before running it. The environment he reports is:

```dotenv
MODEL_DIR=<local path of the downloaded weights>
PORT=8888
HOST=127.0.0.1
UI=no
VISION=auto
CONTEXT_SIZE=262144
CACHE_QUANT=4
GPU_MEM_GB=22.8
CPU_CACHE_GB=0
DRAFT=mtp
PROFILE=4.0bpw-262k
HF_TARGET_REPO=0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw
HF_REVISION=11447639033dd80a4c04536b5b37ed7d642802ee
MODEL_ID=thinkingcap-27b-kearuga-exl3-4.0bpw
SETUP=console
```

After following the kit's installation instructions, his reported launch is:

```bash
# From the kit checkout, on the 5090 host:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 bash linux/start.sh --no-harness --no-pick
```

From this recipe checkout, the public sampled C1 probe is:

```bash
python3 scripts/probe.py --endpoint http://127.0.0.1:8888 \
  --model thinkingcap-27b-kearuga-exl3-4.0bpw --requests 16 --concurrency 1 \
  --thinking on --output results/rtx5090-c1-sampled.json
```

Use a new path and add `--greedy` for a greedy cell. **Do not use the TensorFold Dockerfile/launcher or `--tandem-raw` for this kit.** It serves one request at a time: C4/C8/C16 were not measured. Probe logs reported no cached prompt tokens, but speed-run cache state is unknown because that log was not kept. The host was not a dedicated benchmark machine; the author records pre-run load and idle-GPU state in the full report.

The RTX 5090 D V2 and Laptop GPU are different 24 GB products. None of the above establishes their fit, cold start, throughput or inference correctness.

## DGX Spark — TandemLLM v0.12.0

Use the [primary recipe](TANDEM.md) rather than reproducing local symlinks or a manual setup-marker bypass. Niklas ran the public community ThinkingCap profile with one slot, 262,144 max context, 32,768 default max output tokens, default effort medium and no external lookup corpus (`install.sh --setup`). He reports linking already-verified target files into the profile directory and the K3 EXL3/K4-head drafter set from its registry entry.

His **raw proxy** added `X-Tandem: raw` and forwarded the body/SSE stream; it introduced an unmeasured loopback hop. The receipt has `usage.tandem.parts.injected=0`; the author's log check found no conductor marker. This is evidence about the **raw model path**, not the default conductor. Prompt cached-token counts are zero; that does not mean every engine prefix/session/cache facility was disabled. He did not separately verify that the raw path applied the probe's requested effort.

**Deviations, not recommendations:** the author reduced the disk margin from 60 to 40 GB, encountered three profile-switch smoke failures and manually set aside a restored setup marker before starting. His disk cache was empty/below its free-disk floor. These facts belong beside the measurement; do not present that sequence as a clean, validated installation procedure. A successful existing engine request does not resolve the setup/rollback failure.

Our service is a distinct [v0.11.4 four-slot adaptation](SPARK-TANDEM.md). Do not transfer its concurrency settings or observed health to the stock v0.12.0 single-slot receipt.

## DGX Spark — legacy TensorFold public screen

The independent run used recipe `55efde5692be479224e79f6125334cb6b8bcce25`, a locally built `thinkingcap-exl3:0.2.0` image, the pinned target and stock DFlash2 checkpoint, 16 lanes, 262,144 context and BF16 KV. See the [legacy instructions](TENSORFOLD.md) and [original report](receipts/niklas-pr-1-report.md) for image ID, commands, cache totals and warm-up.

On the author's 2200 MHz clock-capped Spark, the sampled public C1 receipt gives **60.16 tok/s derived net decode**, numerically within 1% of the historical 60.61 tok/s. Its **aggregate is 52.43 tok/s**, a different metric. The public C4/C8/C16 aggregates are 129.51 / 166.33 / 212.10 tok/s. These are different prompts/runs/sampling conditions from the historical table, **not a replay, equivalence test or demonstrated regression**. In particular, historical C4 used sampling while this public C4 used greedy decoding.

Repeated public prompts produced some prefix-cache hits in the later TensorFold cells. The author reports 121 seconds of cold load and a 47-second first request; that warm-up remains in the readiness receipt, not silently removed. One boot / one run cannot establish boot-to-boot spread.

## Remaining report-only results and suggestions

The following numbers are preserved for completeness, **not promoted to receipt-backed README benchmarks**. Underlying speed/quality/`fastrun` run files are not attached. These are Niklas's reported summaries; we have not independently re-run the methods.

| Method | Author-reported result | Important limit |
| --- | --- | --- |
| Fixed-method decode, thinking off | 5090 137.35; Spark Tandem raw 46.30; Spark TensorFold 50.06 tok/s | Same reported method, different from probe/net-decode definitions; harness built/proven on llama.cpp, not separately validated for these engines |
| Tandem `exl3_fastrun.py`, greedy, thinking off | Prose 44.23; code 92.59; chat 61.86 tok/s | Three repetitions; prose stopped around 406–408 tokens, not 512; caches on in `--no-server` mode |
| RTX 5090 MMLU-Pro | 57.25%, n=400 | Thinking off; seeded sample, not the model's intended thinking-on evaluation |
| RTX 5090 GPQA Diamond | 44.44%, n=198 | Public mirror, thinking off |
| RTX 5090 IFEval | 84.5%, n=200 | Prompt-level strict; thinking off |
| RTX 5090 RULER-lite | 82.5% at 8K; 65.0% at 32K, n=40 each | Limited synthetic tasks, not 262K context qualification |

The full report includes the author's Wilson 95% half-widths, prefill/TTFT summaries, seeds, prompt counts, proxy source and caveats. Caps count as wrong in his quality method. These are **not** a rerun of our fidelity/KL or AIME/MATH suites and must not be used as Tandem task-accuracy claims. We retain the author's proposed documentation sentence in that report; the wording above deliberately avoids calling numerical proximity a successful reproduction.

For a stronger follow-up, retain the fixed-speed/quality/`fastrun` JSONs, effective effort, exact asset revisions, clocks, cache state and power settings; repeat boots/runs; test conductor and raw separately; and qualify long contexts/concurrency on each actual hardware SKU. These are suggestions, not claims that such testing has been performed.
