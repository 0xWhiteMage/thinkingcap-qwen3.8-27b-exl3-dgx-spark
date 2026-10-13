# 24 GB NVIDIA GPUs: experimental qualification path

> **Not yet a working, qualified 24 GB deployment.** The pinned runtime can build for Linux x86-64, but its default memory admission refuses this checkpoint on an ideal 24 GiB card. The opt-in launcher below is a guarded qualification attempt, not a claim that an RTX 3090 or RTX 4090 has passed cold start or inference. No 24 GB card was available for that measurement.

The published weight shards occupy 16.35 GB on disk. That is not the loading peak, allocated weight memory, KV cache or total GPU requirement. The Spark's 128 GB unified-memory results must not be presented as 24 GB card speeds or capacity.

## What is different from the Spark profile?

| Setting | Spark profile | Experimental `cuda24` profile |
| --- | --- | --- |
| Host / build | Linux ARM64 / GB10 | Linux x86-64, including a properly configured WSL2 NVIDIA setup |
| GPU selection | Spark GPU | One explicitly selected GPU, `--gpu 0` by default |
| Concurrency | 16 lanes | 1 lane only |
| Context | 262,144 tokens | At most 8,192 tokens |
| Server output default | 131,072 tokens | 4,096 tokens, including reasoning; not a hard per-request ceiling |
| Target KV | BF16 | BF16 |
| Drafter | Pinned stock DFlash 2, 4-bit linears | None; no drafter downloaded or mounted |
| Copy/speculative drafts | Runtime defaults | `--no-drafts --drafter none` |
| Prompt precision | BF16 | BF16; `--no-prefill-fp8` |
| Memory reserve | Upstream default | 2 GiB upstream override, for qualification only |
| CPU affinity | Spark-specific core allocation | No Spark core numbers assumed |
| Evidence | Historical Spark measurements and resident-runtime API smoke | Header/source estimates and x86 image build; no measured 24 GB fit |

## Why default startup is blocked

The [reproducible header receipts](../evidence/24gb-header-estimates.json) were computed from **all 3,080 tensor headers** with the exact installed runtime's `estimate_weights`, `gdn_geometry` and EXL3 admission functions. No model weights were allocated and no GPU was exposed during those estimates.

| Context | Cache/workspace estimate | Serving peak estimate | Loading/startup peak estimate |
| ---: | ---: | ---: | ---: |
| 2,048 | 1.772 GiB | 16.566 GiB | 21.898 GiB |
| 4,096 | 1.945 GiB | 16.739 GiB | 21.898 GiB |
| 8,192 | 2.291 GiB | 17.085 GiB | 21.898 GiB |

Components are 14.794 GiB resident weights plus **7.104 GiB estimated loading staging**. Staging is three times the largest transformed group, dominated by the BF16 embedding tensor. It is an upstream conservative estimate, not a measured allocation trace.

An ideal 24 GiB card receives at most **20 GiB** under TensorFold's default 4 GiB reserve. The exact upstream plan returns `fitting=0` at all three contexts above. **Reducing context does not remove this context-independent loading gate.**

The source permits `TENSORFOLD_MEMORY_RESERVE_GIB=2`. At an ideal 22 GiB budget, the 8,192-token plan passes arithmetic admission. But that requires at least **23.898 GiB actually free after CUDA initialisation**: only about **104 MiB** of prior use is affordable on an exact 24 GiB card. Driver allocations, a desktop display, another process, or a driver-reported total below that threshold can prevent admission entirely. This is too narrow to call a conservative production configuration.

The launcher uses the upstream live budget and host staging checks, not the advertised card size or a rounded `nvidia-smi` total. It does **not** patch out the admission guard or reduce the reserve below the upstream minimum. A refusal is the expected safe outcome on many 24 GB systems.

## RTX 3090 versus RTX 4090

- **RTX 3090 / Ampere (SM 8.6):** the pinned runtime's default CUDA floor is SM 8.9. The profile sets the source-supported `TENSORFOLD_MIN_CAPABILITY=8.0` before imports, allowing its EXL3/W4A16 path. This does not enable unsupported FP8 or NVFP4 tensor-core operations.
- **RTX 4090 / Ada (SM 8.9):** satisfies the stock capability floor, but has the same loading-admission problem. A newer GPU generation does not make the startup bytes disappear.
- The published recipe contains no RTX throughput, quality or measured peak-memory claim. A compiler check for an architecture is not a GPU execution test.

The exact EXL3 linear extension compiled, linked and loaded on the CPU build host for **both SM 8.6 and SM 8.9**; `cuobjdump` confirmed both cubins in the resulting library. See the [build/compiler receipt](../evidence/x86-build-and-kernels.json). This does not test GPU kernel execution or the rest of a real-card cold start.

## Build and inspect without starting a server

Use the main README's clone/venv instructions, then download **only the target**:

```bash
python scripts/artifacts.py --role target
python scripts/artifacts.py --role target --verify-only

docker build --platform linux/amd64 -t thinkingcap-exl3:0.2.0 .

python scripts/serve.py --profile cuda24 --gpu 0 --dry-run
```

The same pinned NVIDIA PyTorch image index contains fixed ARM64 and AMD64 child manifests. The runtime source and model revision are unchanged. Keep enough host RAM for the estimated 7.104 GiB staging allocation **plus** host reserve, Python, Docker and compiler overhead; 32 GB host RAM is a practical starting point, not a measured minimum guarantee. Leave at least the main README's suggested 60 GB free disk headroom.

To reproduce the idealised header estimate without GPU allocation:

```bash
docker run --rm --entrypoint python3 \
  -v "$PWD/models/target:/models/target:ro" \
  -v "$PWD/scripts/capacity_check.py:/capacity_check.py:ro" \
  thinkingcap-exl3:0.2.0 /capacity_check.py \
  --model /models/target --context 8192 --total-gib 24 --budget-gib 20
```

Changing the last budget to `22` reproduces the arithmetic override scenario. Neither command establishes the memory actually available on your card.

## Explicitly opt into a guarded attempt

Only on an idle GPU you control, with no other model resident:

```bash
python scripts/serve.py --profile cuda24 --experimental-24gb \
  --gpu 0 --port 18888 --name thinkingcap-exl3-24gb
```

Before starting the server, an ephemeral container:

1. exposes exactly the selected GPU and reads its free memory **after CUDA initialisation**;
2. reads the verified target's tensor headers, without allocating weights;
3. uses the pinned runtime's actual admission geometry and 2 GiB reserve;
4. checks available host staging memory;
5. refuses the attempt if the requested context cannot pass.

The server repeats its own upstream admission checks during actual loading. A successful preflight is not a guarantee that compilation, calibration, cold start or a generation will fit; available memory can also change between containers. A failed attempt retains the named server container, if one was created, for inspection. The launcher never stops another process or replaces an existing container.

## Requests and qualification after a successful cold start

The server's `--no-drafts` switch disables speculative generation, **not all request-level prefix retention**. Send `"draft": false` on every request to avoid retaining prefixes in this bounded profile. Use a small serial request first:

```bash
curl -N http://127.0.0.1:18888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8 27B Kearuga (ThinkingCap)",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "draft": false,
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "xhigh"}
  }'
```

For real receipts, the public probe also has an explicit no-retention option:

```bash
python scripts/probe.py --endpoint http://127.0.0.1:18888 \
  --no-draft --thinking off --greedy --max-tokens 64 \
  --output results/24gb-ready.json

python scripts/probe.py --endpoint http://127.0.0.1:18888 \
  --no-draft --thinking on --requests 4 --concurrency 1 --max-tokens 1024 \
  --output results/24gb-thinking-serial.json
```

Keep prompt plus generated tokens within the enforced 8,192-token window. The launcher limits its configured output default to 4,096, but TensorFold lets individual requests override that default through `max_tokens` or `max_completion_tokens`; it is not an enforced server-side output ceiling. For this qualification profile, keep each requested output budget at or below 4,096 and reduce it further when the prompt leaves less room. Do not send the main README's Spark C4 concurrency or 8,192-output example unchanged. A requested 4,096-token thinking budget can truncate difficult tasks; a `length` finish reason is not a completed answer or a quality pass.

Before claiming a card supported, record GPU/driver identity, free memory at startup, actual cold-start success, observed peak allocation during loading and prefill, serial short and near-window requests, final answers, usage, finish reasons, and repeat-run stability. Keep those receipts separate from the historical Spark benchmarks.

**Until that evidence exists, the accurate status is experimental and potentially blocked by startup admission—not “runs on any 24 GB GPU”.**
