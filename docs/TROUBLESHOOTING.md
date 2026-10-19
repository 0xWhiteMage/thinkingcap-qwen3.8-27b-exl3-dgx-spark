# Troubleshooting

This page covers the **legacy TensorFold Docker recipe and experimental cuda24 profile**. For the primary serving route, use [the Tandem guide](TANDEM.md), including setup/rollback, raw/conductor and rendered-launcher checks.

## Build fails with an architecture or CUDA-version error

Build the Spark profile on Linux aarch64 with `--platform linux/arm64`; use `--platform linux/amd64` on the experimental discrete-GPU path. The pinned NVIDIA image index selects fixed child manifests for those two architectures, not a moving tag. Do not substitute another base and retain this recipe's qualification label. An NGC access or network error is a download blocker, not a model failure.

## A 24 GB card is refused despite the 16.35 GB weight download

This is a known startup-admission limitation, not proof of corrupt weights. The exact header-based estimate is 21.898 GiB at loading; the default budget on an ideal 24 GiB card is at most 20 GiB. Lower context cannot bypass the loading gate. The experimental profile uses the supported 2 GiB reserve but needs at least 23.898 GiB free after CUDA initialisation and can still refuse. Do not patch out the guard or claim the image build proves a measured fit. Read the [24 GB qualification guide](24GB-GPUS.md).

## Patch checksum or source-tree check fails

Use a clean clone of the recipe at your recorded commit. Do not edit source or disable the checks to force the build through. The first patch must reconstruct tree `0c59cc48df1ad49957ce7e56d039f205d905f7f3` from the pinned Python base before the GB10 patch. PR #260's public final head and its local rebased commit are different identifiers.

## EXL3 cannot be loaded

This is EXL3, not NVFP4, FP8, GGUF or BF16. Use the image built by this Dockerfile. Native TensorFold 1.x and the SGLang recipe are different runtimes; their commands and weight formats are not a fallback for this pack. Preserve the tokenizer and configuration from the pinned release rather than borrowing files from base Qwen.

## Model checksum or file-size mismatch

Run `python3 scripts/artifacts.py --verify-only` and identify the exact failing path. Download that snapshot again through the pinned `hf download` route. Never rewrite the manifest to accept a corrupt file. Mirrors must contain the same bytes; check them with the same manifest. LFS pointer text is not the model payload and will fail verification.

## The launcher refuses an existing container or occupied port

This is intentional. It will not delete a container, terminate another model or change SparkDash. Inspect the owner of the service first. For this recipe's own stopped container, explicitly `docker rm thinkingcap-exl3` before recreating it. To resume without changing settings, use `docker start thinkingcap-exl3`. Changing the port alone is not permission to load a second large model into shared GPU memory.

## First startup takes a long time

Inspect `docker logs -f thinkingcap-exl3`. CUDA extensions compile and the planner calibrates on first use. The host-mounted `cache/` directory retains Torch/Triton compilation artefacts. Fresh caches and different image/source versions can require new builds; the local resident cache from a different installation is not a required recipe download. Readiness is not proof of all capacity settings or task accuracy.

## No final answer, only reasoning

Inspect `finish_reason`, `capped` and `has_final_answer` in the receipt. `max_tokens` covers reasoning and final tokens together; reaching `length` is not a complete answer. Increase the budget on a diagnostic request, or use thinking off for a short API smoke. Do not claim that a speed-probe output cap establishes reasoning quality.

## Probe fails or aggregate throughput is null

Check the recorded error and endpoint identity. A network/API failure, missing usage, incomplete SSE stream or server error is not zero throughput or a wrong model answer. Fix the endpoint or client compatibility and run with a new output path. The probe expects the pinned server's OpenAI-style stream, terminal `[DONE]` and final usage; it is not a generic evaluator for every provider.

## Throughput differs from the model card

Match the target/drafter revisions, source patches, architecture, BF16 target KV, sampling/thinking settings and concurrent workload first. Record driver, clocks, prompt-cache state, prompt length, output length and background load. This recipe deliberately does not change GPU clocks or stop unrelated work. The public prompt screen is not the unpublished historical battery, and its metric includes wall-time overhead. Do not compare its C1 aggregate rate with historical C1 net decode.

## Remote clients cannot connect

The default is localhost only. For a trusted private network, the launcher supports `--host 0.0.0.0`, but does not configure authentication or a firewall. Configure access restrictions and an authenticated proxy before exposure. Never publish an unauthenticated port or embed API credentials in the README. `RECIPE_API_KEY` lets the probe authenticate to a proxy without printing the key in its receipt; it does not configure server authentication.

## Long contexts run out of memory

Weight size is not total memory. KV, drafter, saved prompt states and runtime buffers grow with workload. The 262,144 context setting limits prompt + reply per request; it is not a promise of 16 maximum-length sessions. Lower `--parallel` or `--context` for your workload, label the changed configuration, and re-evaluate before claiming equivalence to the historical serving profile.
