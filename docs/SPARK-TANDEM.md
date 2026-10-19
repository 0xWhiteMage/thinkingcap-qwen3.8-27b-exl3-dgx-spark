# How our Spark currently serves ThinkingCap through Tandem

[Primary public recipe](TANDEM.md) · [Independent v0.12.0 receipts](RECOMMENDED-RECIPES.md) · [Sanitised inspection evidence](../evidence/tandem-inspection.json)

This is an inspection of the **already-resident service on 19 October 2026**, not a fresh installation recipe or a new benchmark. No service was stopped, upgraded or restarted, no profile was switched, and no second large model was loaded.

## Observed topology

```text
OpenAI-compatible client → :8000/v1/chat/completions
                         │
                         ├─ default: conductor on → routing/packages/effort policy/monitor
                         └─ X-Tandem: raw → plain model path
                                             │
                                  TandemLLM v0.11.4
                                  ThinkingCap Kearuga EXL3 4.00bpw
                                  Tandem K3 EXL3 drafters + K4 draft head
                                             │
                                  1 × 262,144-token main slot
                                  3 ×   8,192-token side slots
```

| Item | Observed value / boundary |
| --- | --- |
| Engine source | Clean upstream checkout, commit `760136f7a18fe33bf0a5fa2439793ba313686379` (`v0.11.4`) |
| Public API model ID | `TandemLLM` |
| Active profile | `exl3-4.0`, with a local target-directory symlink to ThinkingCap |
| Target identity | Both loaded-path weight shard sizes/SHA-256 values match [the historical target manifest](../manifests/target.json) |
| Default effort / output | `medium` / 32,768 completion tokens; the probe explicitly supplies its own thinking/sampling fields |
| Reasoning format | `tags`; on this version raw responses can leave `<think>…</think>` in `content` |
| Conductor | `on`; default requests can gain package tokens and a policy-selected effort |
| Lookup corpus | Empty external corpus setting; conversation lookup is still possible |
| Real concurrency | Four slots confirmed by health; not inferred from queue depth |
| Memory guard | `QSE_PARALLEL_FLOOR_GB=15` |
| Waiting queue | `MAX_QUEUE=16`, not 16 simultaneous decoders |
| Driver / clock observation | GB10, driver `580.173.02`; one `nvidia-smi` snapshot reported 2418 MHz application graphics / 2411 MHz current graphics. Not a controlled benchmark-clock or power measurement |
| Network | Existing service binds all interfaces on port 8000; this inspection did not change it. The public recipe recommends localhost plus a tunnel/authenticated proxy |

## Two important local adaptations

### 1. Profile name is not proof of target identity

The installed profile is the stock `exl3-4.0` route, redirected to ThinkingCap through a local symlink. Its health metadata still names the stock `0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.0` HF repository. That metadata is **not evidence that the stock target is loaded**. We resolved the actual weights location and hashed both target shards; they match the Kearuga manifest.

This establishes the target tensor-file identity, not complete equivalence of tokenizer, template, base/plain tensors, drafters or dependency closure. The local profile retains supporting base/drafter assets and its original NVFP4 fallback paths. Those command-line paths do not imply that the target EXL3 tensors are NVFP4 or that another large engine is concurrently loaded.

New public installations should use the explicit **community ThinkingCap profile** in [the Tandem guide](TANDEM.md), not rely on our older stock-profile symlink convention.

### 2. The rendered launcher actually forwards the slot settings

The installation separates clean upstream `src/` from machine-specific `run/ops/serve.env` and `run/ops/start.sh`. Our rendered launcher adds:

```bash
--parallel "${QSE_PARALLEL:-1}" --side-max-len "${QSE_SIDE_MAX_LEN:-65536}"
```

With `QSE_PARALLEL=4` and `QSE_SIDE_MAX_LEN=8192`, health reports one 262,144-token main slot plus three 8,192-token side slots. A request too long for a side window must wait for the main slot. Four users do **not** each receive a guaranteed 262K window.

The inspected public v0.12.0 `ops/start.sh` does not forward those CLI arguments. Therefore copying `QSE_PARALLEL=4` into that version's environment alone does not reproduce our service. The public recipe intentionally starts with stock one-slot behaviour. Do not describe the independent one-slot C1 receipt as a four-slot benchmark or copy the historical TensorFold 16-lane setting into Tandem.

## What the live API checks established

Six short public-prompt requests were sent sequentially: one thinking-off readiness request and two thinking-on requests on each path. All six returned final content, `finish_reason=stop`, **zero errors and zero cap hits**. Both thinking-off readiness replies were exactly `READY`.

- **Raw path:** `usage.tandem.parts.injected=0`, no harness. Thinking-on usage reported non-zero reasoning-token counts. Tagged reasoning can remain inline on v0.11.4.
- **Default path:** injected tokens and harness metadata were present. The two thinking-on responses reported the conductor's `medium` effort policy; sending nested `xhigh` is not enough to claim both paths used the same effective policy.
- These checks verify transport, completed short answers and the practical raw/conductor distinction. They do **not** qualify accuracy, long-context behaviour, sustained concurrency, comparative speed or a fresh v0.12.0 deployment.

The [published inspection JSON](../evidence/tandem-inspection.json) contains settings, hashes, usage counts and finish states, with private reasoning and non-readiness answer text omitted. Small smoke-request timings are deliberately not promoted as benchmarks.

## Candidate recipe checks without a deployment

The pinned v0.12.0 installer was read and its **`--setup --no-start --dry-run`** path was exercised on this GB10. It exited successfully, checked aarch64/SM 12.1/CUDA 13.0, and printed the pinned clone/dependency/configuration plan. The unused candidate install directory did not exist afterwards. That is a plan/preflight check, **not** a v0.12.0 install, load or inference pass.

A real version/profile migration needs an explicit maintenance window, backup and rollback, source/asset checks, a fresh load, and raw/default-path smoke requests. This documentation update does not perform that live change.
