# Benchmark methodology and claim boundaries

## Three distinct kinds of evidence

1. **Historical release qualification.** `evidence/source/BENCHMARKS.json` and `RELEASE_PROVENANCE.json` are copied unchanged from the pinned target release. They are the source for the README's performance, fidelity and reasoning-task tables.
2. **Recipe verification.** A fresh ARM64 Docker build, source-text comparison and small live API probes establish that the build is usable and the client can talk to the resident qualified stack. They do not repeat the full qualification.
3. **Upstream ThinkingCap context.** BottleCap's BF16 model card owns the 37.2% macro-average reasoning-token reduction. It is not an EXL3 result, and it cannot be multiplied into a serving-stack throughput gain.

## Historical speed measurements

The baseline and current arms use the same ThinkingCap EXL3 target and stock DFlash 2 checkpoint on one DGX Spark / GB10, with BF16 target KV, a 262,144-token context setting and a 16-lane serving limit. The comparison changes the serving stack, not the model or drafter.

- **C1/C4:** a 50-prompt battery across five domains; two runs per boot, two boots per stack. Thinking enabled, `reasoning_effort=xhigh`, model-default sampling, 1,024-token output cap.
- **C1 net decode:** completion tokens divided by decode time after first token, excluding TTFT. This is not full end-to-end throughput.
- **Concurrent aggregate:** all cell completion tokens divided by the cell's total wall time. It includes request overhead and is not a per-user decode rate.
- **C8/C16:** greedy descriptive screen, not a powered claim of universal superiority.
- **Cold long-prompt TTFT:** median measured first-token wait on the recorded cold prompt workload, not a universal startup latency.

Output includes reasoning. Capped requests are speed probes; they are not proof of correctly completed tasks. Changing prompts, prompt caches, output length, sampling, load or clocks changes the result. This is not an EXL3/BF16 or TensorFold/SGLang comparison.

## Historical fidelity

The internal G3-v2 suite has 240 prompts and 25,780 teacher-forced positions. It compares the served EXL3 model against BottleCap's ThinkingCap BF16 revision `4bd4e11054e4ceb0dcecfff2f0d2ffa906f37232`.

Each position uses frozen reference top-20 token support plus an other-token bucket. Mean and tail KL are therefore **coarsened KL**, not full-vocabulary KL. Top-1 agreement measures teacher-forced token positions, not free-running answer equivalence. The internal prompts and captures are not redistributed here; a numerical replay of these fidelity results is not provided by this recipe.

## Historical reasoning-task timing

Hard-V2-R contains all 30 AIME 2025 problems and a fixed 40-problem MATH-500 Level-5 sample, repeated three times, at concurrency four with thinking `xhigh` and model-default sampling.

AIME outputs are capped at 24,576 tokens; MATH outputs at 16,384. Reported means are the arithmetic mean of the three existing per-run means. Incorrect and capped attempts stay in the averages. They are actual attempt wall times, not estimates from tok/s and not successful-only completion times.

The recorded quality verdict is registered non-inferiority. The 62/70 versus 60/70 averages do not establish general superiority. The current AIME cap-hit counts are 6, 6 and 5; MATH has no cap hits. Timing changes are descriptive, not a separately qualified latency win.

## Public probe in this repository

`bench/prompts.json` is a new ten-prompt diagnostic set. It is **not** the internal 50-prompt battery, G3-v2 or Hard-V2-R. The first prompt asks for `READY`; subsequent prompts cover code, math, instructions, text and tool boundaries. There is no automatic task-accuracy score.

`scripts/probe.py` records:

- Client-observed TTFT to the first non-empty reasoning or answer text event. This includes connection/header latency; it is not an internal engine prefill timer.
- Request wall time, server usage, content, reasoning content and finish reason.
- Cap hits and whether final content exists; these are not silently discarded.
- Aggregate completion tokens divided by the entire public probe wall time, including executor and connection overhead.

It **does not calculate C1 net decode** or count one SSE event as one token. If any request errors, aggregate throughput is null and the process exits non-zero. Missing usage or an incomplete stream is an error, never a zero-token success. A capped reply can still contribute tokens to a speed probe but is not treated as a completed answer.

For a descriptive public concurrency screen, run each cell separately on an otherwise idle server with a new receipt path:

```bash
python scripts/probe.py --requests 16 --concurrency 1 --thinking on --greedy --output results/public-c1.json
python scripts/probe.py --requests 16 --concurrency 4 --thinking on --greedy --output results/public-c4.json
python scripts/probe.py --requests 16 --concurrency 8 --thinking on --greedy --output results/public-c8.json
python scripts/probe.py --requests 16 --concurrency 16 --thinking on --greedy --output results/public-c16.json
```

The ten prompts repeat when requests exceed ten. Prefix caching and repeated prompts can affect these cells; record warm/cold state, recipe commit, image ID, driver, clocks and competing load with your results. Do not label this screen as a replay of the published historical table. No benchmark is run automatically during installation.
