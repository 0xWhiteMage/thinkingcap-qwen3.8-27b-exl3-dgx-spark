## Summary
Following your DM about the RTX 5090 and a PR with benchmarks and setup, this PR starts there. It adds plain receipts and the setups behind them. No code changes.

1. **RTX 5090 32 GB** (new): the same Kearuga EXL3 4.00bpw weights, run through ExLlamaV3 1.6.0 + MTP via a community kit. This is **not** your TensorFold recipe, which is qualified on the Spark only (it also has an experimental 24 GB profile, which I did not run). I ran your `scripts/probe.py` unchanged, plus a fixed speed method and a small quality set.
2. **DGX Spark-class unit with TandemLLM v0.12.0** (the engine you said you run it with): re-measured independently with your `probe.py` and with TandemLLM's own `exl3_fastrun.py`. The benchmark requests went through a small pass-through proxy that adds the header `X-Tandem: raw`. What it does and why is explained below; please read that section.
3. **Same Spark unit with your recipe** (TensorFold, `55efde56`): the probe cells from `docs/BENCHMARKS.md`, as an addition.

The numbers are descriptive. They are not a replay of your historical table, as your docs say. Limits are listed in "What this does not show".

Credit line, if you use these numbers: **Benchmarks by Niklas Lenz (@niklaslenz_ai), niklaslenz.com**

---

## 1. RTX 5090 32 GB

### Setup
| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| Driver / kernel | 610.43.02 / 6.12.95+deb13-amd64 |
| Power / clock | Power limit as configured on the host, not changed for these runs. No clock cap. |
| Engine | ExLlamaV3 1.6.0 through the MiaAI-Lab one-click kit (`MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install` @ `bbb11db`), single-request (batch 1) server, MTP drafting (`DRAFT=mtp`) |
| Weights | `0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw` @ `11447639033dd80a4c04536b5b37ed7d642802ee`, 26/26 files verified against `SHA256SUMS` |
| Context / KV | 262,144 / int4 KV (`CACHE_QUANT=4`), kit GPU budget 22.8 GB |
| Probe | `scripts/probe.py` and `bench/prompts.json` from your recipe @ `55efde5692be479224e79f6125334cb6b8bcce25` (`probe.py` sha256 `5438815a422510afa701e0b65c892fe5f180f802614af70c17f33ce5cbdf2628`, `prompts.json` sha256 `4c46506e7696f508564d1f6f9ec0ac72fd43ce07cbd5948f700c3caa9c911e34`), unchanged |
| Sampling / thinking | `probe.py` sends 1.0 / 0.95 / 20 (or temperature 0 with `--greedy`) and `enable_thinking` true. It also sends `reasoning_effort=xhigh`, but the kit server reads only `enable_thinking` from `chat_template_kwargs`, so effort is the template default, which is xhigh. |
| Prompt cache | The kit does not report cached tokens in `usage`. Its own log shows no cached prompt tokens in any of the 33 probe requests (smoke + 16 + 16). The log of the speed run was not kept, so the cache state there is unknown, not zero. |
| Host state | GPU idle before the start (2 MiB used, no compute process). The host is not a dedicated benchmark machine; its 1-minute load average was 4.83 when the run started. |

Kit `.env` used:
```
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
Start, then the three probe calls (one cell per run, new receipt path each):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 bash linux/start.sh --no-harness --no-pick
M=thinkingcap-27b-kearuga-exl3-4.0bpw
python3 scripts/probe.py --model $M --thinking off --greedy --max-tokens 64 --output smoke-ready.json
python3 scripts/probe.py --model $M --requests 16 --concurrency 1 --thinking on --output c1-sampled.json
python3 scripts/probe.py --model $M --requests 16 --concurrency 1 --thinking on --greedy --output c1-greedy.json
```

### Results, `scripts/probe.py` (16 requests per cell, `--thinking on`, `max_tokens` 1024, 0 errors)
| Cell | Probe aggregate (tokens / wall) | Derived net decode (sum of completion tokens / sum of (wall - TTFT)) | Completion tokens | Mean TTFT | Cap hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1, model-default sampling (no `--greedy`) | 151.10 tok/s | 161.74 tok/s | 3,370 | 0.092 s | 1 |
| C1, `--greedy` | 164.08 tok/s | 178.92 tok/s | 2,789 | 0.088 s | 0 |
| C4 / C8 / C16 | not measured | not measured | | | |

- C4 / C8 / C16 are not measurable here: the kit serves one request at a time.
- The one cap hit is the `code-race` prompt in the sampled cell: `finish_reason` length at 1,024 tokens, no final answer. Its 1,024 tokens are inside the 3,370 and the aggregate, as your docs describe for capped speed probes. It is not a completed answer.
- `probe.py` does not compute net decode (your `docs/BENCHMARKS.md`). I derived it from the receipt as sum of completion tokens divided by sum of (wall - TTFT). Do not compare the aggregate with a net decode.
- The 5090 receipts have no `completion_tokens_details` in `usage`: the kit reports no `reasoning_tokens` at all, although `reasoning_content` is filled in all 16 records of both cells. Completion tokens therefore include reasoning, as in your definition, but the reasoning share cannot be read from these receipts.
- Mean TTFT is the client-observed time to the first reasoning or answer text, as `probe.py` records it.
- `smoke-ready.json` is only a readiness check (2 tokens).

### Results, fixed speed method (thinking off)
One warm-up, then two fixed prompts (code, prose) x two runs x 512 tokens, median of the per-prompt medians, seed 42, no prompt cache requested (`cache_prompt` false), card sampling 1.0 / 0.95 / 20 / min_p 0.0, thinking off. The tool is a copy of `alexellis/rigmark` 218c8ae (protocol 1.3.0), extended.

| Metric | RTX 5090 |
| --- | ---: |
| Decode | 137.35 tok/s (CV 3.3 %) |
| Prefill, 2K / 8K / 32K prompt | 3241.92 / 3322.09 / 3029.34 tok/s |
| TTFT, 2K prompt | 0.632 s |
| Speculative acceptance | not recorded by my speed tool (the kit logs it per request) |

The speed receipts (`speed2.json`) are not attached because they contain local paths. The values above are the summary line of the run.

### Quality, my method (thinking off, not your battery)
Fixed lists, seed 42, the margin is the Wilson 95 % half-width in percentage points. Answers cut at the token limit count as wrong. MMLU-Pro: 400 questions sampled with seed 42. GPQA Diamond: all 198 questions from the public mirror (the original is gated). IFEval: 200 prompts sampled with seed 42, prompt-level strict. RULER-lite: 4 tasks x 10 items per length.

| Test | n | Result |
| --- | ---: | ---: |
| MMLU-Pro | 400 | 57.25 % (+-4.83) |
| GPQA Diamond | 198 | 44.44 % (+-6.86) |
| IFEval | 200 | 84.5 % (+-5.01) |
| RULER-lite 8K | 40 | 82.5 % (+-11.6) |
| RULER-lite 32K | 40 | 65.0 % (+-14.18) |

ThinkingCap is meant to run with thinking on. These runs had thinking **off**, so they say little about the model as you serve it. They are not comparable with your AIME / MATH timing or your KL numbers. They are included for completeness.

---

## 2. DGX Spark with TandemLLM v0.12.0

### Setup
| Item | Value |
| --- | --- |
| Hardware | One GB10 unit, 128 GB unified memory (DGX Spark class; the unit is an ASUS Ascent GX10) |
| Driver / kernel | 580.173.02 / 6.17.0-1031-nvidia |
| Clock | GPU clock capped at 2200 MHz (a fixed setting on my side). Power limit not changed. |
| Engine | TandemLLM v0.12.0 (by 0xBakeer; tag `v0.12.0`, commit `f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf`), installed with `install.sh --setup` (torch 2.13.0+cu130) |
| Profile | `community/thinkingcap-27b-kearuga-exl3-4.00` from the community registry (hf revision `11447639`, drafters `tandem-exl3`: K3 EXL3 drafters + K4 draft head, per the registry entry) |
| Weights | `0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw`. I linked the files of your recipe pin `d6df32fa` into the profile's weights directory; they are byte-identical to HF `11447639` (only `README.md`, `FILES.json` and `SHA256SUMS` differ between the two revisions). |
| Server settings | TandemLLM defaults from `ops/serve.env`: one slot (`QSE_PARALLEL=1`), `MAX_LEN` 262,144, default max tokens 32,768, default effort `REASONING_EFFORT=medium`, conductor on (benchmark requests use the raw path, see below). `probe.py` sends `reasoning_effort=xhigh` in every request; TandemLLM's docs say a client's own effort word wins over the server default (`docs/benchmarks.md`, "Settings that change the score"). I did not verify separately that the raw path applies the request value. |
| Probe | Your `probe.py` @ `55efde56`, unchanged, pointed at the proxy port |
| Prompt cache | Probe receipts show `cached_tokens` 0 for every request. The engine's own prefix and session cache stayed at its defaults. |
| Host state | Idle: no other server, container or GPU process running |

Deviations from a plain install, stated openly:
- `ops/use-profile.sh` refused to switch (it wants 60 GB free disk margin; the unit had 54 GB free, from my run notes). I ran it with `DISK_MARGIN_GB=40` (the script's own variable). Nothing else in the profile or the server settings was changed. The engine's disk cache is on but empty and limited: free disk (44 GB) is below its 50 GB floor, 0 entries. The benchmark cells do not use it.
- `use-profile.sh`'s own smoke step failed in three runs (twice with an empty `curl` answer, once with "/health does not name the profile" while the box was still in setup mode). The same request sent by hand worked. After the failures the installer's `run/setup-pending` marker was back; I set it aside by hand so that `ops/start.sh` starts the engine with the profile instead of setup mode. I did not chase the cause of the failed smoke step.
- I installed with `install.sh --setup`, which builds no lookup corpus (`CORPUS=''` in `ops/serve.env`). TandemLLM's README says the lookup drafter then proposes different continuations, so speeds can differ from the paper's numbers; it says it has not measured by how much, and neither have I.

Commands:
```bash
ops/use-profile.sh community/thinkingcap-27b-kearuga-exl3-4.00    # with DISK_MARGIN_GB=40
ops/start.sh                                                      # engine on :8000
python3 tandem_rawproxy.py 8001                                   # pass-through proxy, source below
python3 scripts/probe.py --model TandemLLM --endpoint http://127.0.0.1:8001 \
  --requests 16 --concurrency 1 --thinking on --output sampled-c1-tandem.json
python3 tools/exl3_fastrun.py --repo <TandemLLM checkout> --no-server --port 8001 --served-model TandemLLM \
  --workloads prose,code,chat --prefill "" --label tandem-0.12.0-thinkingcap --json fastrun-greedy.json   # TandemLLM's own tool; defaults: 512 new tokens, 3 reps, temperature 0
```
(Both calls are written down from the result files and the setup, not copied from a shell history: the receipt settings give model, request count, concurrency, `max_tokens` and thinking; the endpoint is the proxy port. For `fastrun`, the result file gives label, served-model name, workloads, 3 repetitions, 512 tokens and an empty prefill part; greedy decoding is the tool's default, and the engine log shows no sampling tag on these requests.)

### The `X-Tandem: raw` proxy: what it does and why
TandemLLM v0.12.0 serves a conductor in front of the model by default: it routes the turn, can inject notes into the prompt, can change the effort and runs a session monitor. TandemLLM's own docs say that a benchmark of the model should run the plain engine path, and that this path is selected by the request header `X-Tandem: raw` (`docs/benchmarks.md`, section "Two paths"; `docs/server.md`, section "TandemLLM and the conductor"). My speed tool and `probe.py` cannot send custom headers. So I put a small proxy in between. It forwards each request unchanged to the engine port and adds one header, `X-Tandem: raw`. Body and SSE stream pass through unbuffered; `Host` and `Content-Length` are rebuilt and hop-by-hop headers are dropped. Without the proxy, these numbers would have measured the conductor path.

How I checked that the header took effect: the engine's per-request log lines for the 16 probe requests carry no `conductor=on` marker (the conductor-path requests in the same log do), and `usage.tandem.parts.injected` is 0 for all 16 records in the receipt. The proxy adds one loopback hop; I did not measure its cost separately.

<details>
<summary>tandem_rawproxy.py (sha256 8cc9ae014774d7bdc89e44a209a2540f35f2932ac5c9105499e9730d636d1cb1)</summary>

```python
#!/usr/bin/env python3
"""Pass-through proxy 127.0.0.1:8001 -> 127.0.0.1:8000 that adds the header `X-Tandem: raw`
(TandemLLM docs/benchmarks.md:15, docs/server.md:244: a benchmark runs the plain engine, raw is a header).
Nothing else is changed: body, other headers, streaming (SSE) pass through unbuffered."""
import http.client, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP = ("127.0.0.1", 8000)
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
       "proxy-authenticate", "host", "content-length"}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # body ends with the connection: no chunk framing to rebuild

    def log_message(self, *a):
        pass

    def do(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        hdr["X-Tandem"] = "raw"
        if body is not None:
            hdr["Content-Length"] = str(len(body))
        try:
            c = http.client.HTTPConnection(*UP, timeout=3700)
            c.request(self.command, self.path, body=body, headers=hdr)
            r = c.getresponse()
        except Exception as e:  # noqa: BLE001
            self.send_response(502); self.end_headers(); self.wfile.write(str(e).encode()); return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                b = r.read1(65536)
                if not b:
                    break
                self.wfile.write(b); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            c.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8001), H).serve_forever()
```
</details>

### Results, `scripts/probe.py` (16 requests, `--thinking on`, `max_tokens` 1024, 0 errors, 0 cap hits)
| Cell | Probe aggregate (tokens / wall) | Derived net decode (sum of completion tokens / sum of (wall - TTFT)) | Completion tokens | Mean TTFT | Your card (TensorFold stack) |
| --- | ---: | ---: | ---: | ---: | --- |
| C1, model-default sampling (no `--greedy`) | 62.22 tok/s | 66.02 tok/s | 2,726 | 0.158 s | C1 net decode 60.61 |

This is a different engine, a different drafter set and a different KV path than your card, so the card column is context, not a like-for-like target. The sampled runs of different engines also generate different text (2,726 tokens here, 3,306 on the TensorFold run below), so tok/s between them is not an identical-workload comparison.

### Results, TandemLLM's own `tools/exl3_fastrun.py` (greedy, 512 new tokens, thinking off, 3 runs, median)
| Kind | My run, v0.12.0 | The three runs | TandemLLM README / registry entry (v0.8.2) |
| --- | ---: | --- | ---: |
| Prose | 44.23 tok/s | 44.23 / 42.91 / 44.26 | 39.6 |
| Code | 92.59 tok/s | 92.24 / 92.70 / 92.59 | 95.9 |
| Chat | 61.86 tok/s | 61.80 / 61.86 / 61.88 | 51.3 |

- The README and registry figures were measured on 0.8.2 with one run per kind; mine are on 0.12.0 with three runs. The prose answers ended at 406 to 408 tokens (`stop`), not at 512. The README names no clock cap; my unit has the 2200 MHz cap.
- The prose, code and chat prompts are the tool's own; I did not change them.
- `--no-server` measured the installed engine, which runs with its caches on; the tool's own server mode starts the engine without caches. The README and the registry entry do not say which mode produced their v0.8.2 figures.

### Results, fixed speed method (thinking off)
Same method and settings as in section 1 (two prompts x two runs x 512 tokens, median, seed 42, card sampling), requests through the raw proxy. Placed next to the same method on your recipe, same unit:

| Setup (same unit, same method) | Decode | CV | Prefill, 2K / 8K / 32K prompt | TTFT, 2K prompt |
| --- | ---: | ---: | ---: | ---: |
| TandemLLM v0.12.0 (raw path) | 46.3 tok/s | 1.8 % | 683.16 / 833.13 / 803.98 tok/s | 2.999 s |
| Your recipe, TensorFold (section 3) | 50.06 tok/s | 9.7 % | 1132.23 / 1286.36 / 1235.96 tok/s | 1.809 s |

This is a measurement of two stacks as installed on this unit (TandemLLM with `--setup`, so no lookup corpus; TensorFold as shipped), not a ranking of the engines. The TandemLLM engine keeps its own prefix cache on by default; the speed receipts show `cached_tokens` 0 in all 14 checked requests.

---

## 3. Same unit with your recipe (TensorFold), as an addition

### Setup
- Recipe `55efde5692be479224e79f6125334cb6b8bcce25`, image `thinkingcap-exl3:0.2.0` built locally from it (ID `1b1b3bb89fb2b75b4cdd94fe3d04b539f18b18cbf7988ffa338722e1cee2479a`, so it differs from your historical image ID).
- Target at the recipe pin `d6df32fa` (28 files verified by the recipe's own check), drafter `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (5 files verified). The target is byte-identical to HF `11447639` apart from `README.md`, `FILES.json` and `SHA256SUMS`.
- Spark profile as shipped (`python scripts/serve.py`, no flags): TensorFold Python CUDA, stock DFlash2 drafter with `--drafter-bits 4`, 16 lanes, 262,144 context, BF16 KV. Server defaults 1.0 / 0.95 / 20, thinking on.
- Same unit, driver, kernel and 2200 MHz clock cap as in section 2. Idle box, nothing else running, `max_tokens` 1024.
- Prompt cache: server default (on). The ten prompts repeat, so some requests report cached prompt tokens in `usage`: none in `public-c1` and the smoke cells; 146 of 1,257 prompt tokens in `public-c4` (2 of 16 requests); 338 in `public-c8` and in `sampled-c1` (4 of 16 requests each); 406 in `public-c16` (5 of 16 requests).
- Probe calls as in `docs/BENCHMARKS.md` (lines 51 to 54, with `--output results/<name>.json`; the block below shows the bare file names), plus a sampled C1 cell (same call without `--greedy`):
```bash
python scripts/probe.py --requests 16 --concurrency 1 --thinking on --greedy --output public-c1.json
python scripts/probe.py --requests 16 --concurrency 4 --thinking on --greedy --output public-c4.json
python scripts/probe.py --requests 16 --concurrency 8 --thinking on --greedy --output public-c8.json
python scripts/probe.py --requests 16 --concurrency 16 --thinking on --greedy --output public-c16.json
python scripts/probe.py --requests 16 --concurrency 1 --thinking on --output sampled-c1.json
```

### Results (16 requests per cell, 0 errors, 0 cap hits, 2,697 completion tokens in each greedy cell, 3,306 in the sampled cell)
| Cell | Probe aggregate (tokens / wall) | Derived: sum of tokens / sum of (wall - TTFT) | Mean TTFT | Your card |
| --- | ---: | ---: | ---: | ---: |
| C1, model-default sampling | 52.43 tok/s | 60.16 tok/s | 0.506 s | C1 net decode 60.61 |
| C1, `--greedy` | 57.49 tok/s | 69.62 tok/s | 0.511 s | n/a |
| C4, `--greedy` | 129.51 tok/s | 38.34 tok/s (not a per-user rate) | 0.627 s | C4 137.59 |
| C8, `--greedy` | 166.33 tok/s | 27.68 tok/s (not a per-user rate) | 0.950 s | C8 214.88 |
| C16, `--greedy` | 212.10 tok/s | 31.56 tok/s (not a per-user rate) | 1.903 s | C16 249.55 |

- **C1 net decode reproduces:** 60.16 against 60.61 tok/s (-0.7 %). It is the only cell that is close to a like-for-like comparison, because the card's C1 figure is a net decode and the derived column matches that definition.
- **C4 / C8 / C16 are 6 % / 23 % / 15 % below your card.** I do not read this as a regression:
  1. The card's C1 and C4 values come from the 50-prompt battery (two runs per boot, two boots per stack, 1,024-token cap, model-default sampling); my C4 cell is greedy. C8 and C16 on the card are described as a greedy descriptive screen. The public probe is ten short prompts repeated to 16 requests, about 169 completion tokens per request in the greedy cells, so wall time contains a large share of ramp-up and tail (`docs/BENCHMARKS.md`, "Public probe in this repository").
  2. The 2200 MHz clock cap on my unit; the card names none.
  3. A locally built image.
  4. For C1 the card is a net decode and the probe aggregate is a different quantity. For C4 to C16 the card is an aggregate as well, but over a different workload. My derived column is not a clean per-user rate at C4 and above.
- Cold start on this unit: 121 s load (cold CUDA build); the first request took 47 s (warm-up, kept in `smoke-ready.json`).
- One boot, one run per cell, so no boot-to-boot spread.

---

## What this does not show
- Task accuracy, fidelity (KL) and AIME / MATH timing for the serving stacks are not repeated and not shown. The quality table in section 1 is my own thinking-off method on the 5090 only.
- The 5090 numbers come from a batch-1 server with a different engine, MTP drafting and int4 KV. They say nothing about your TensorFold recipe, and C4 and above were not measured.
- The 5090 kit reports no `reasoning_tokens`, and one of 32 5090 probe requests hit the token cap (listed above). The 5090 speed and quality runs and the probe cells were separate server starts of the same kit configuration.
- The Spark results with TandemLLM depend on the raw proxy described above. The default conductor path was not measured.
- The 50-prompt battery is internal, so the historical C4 / C8 / C16 cells cannot be replayed one to one.
- All probe cells are one boot, one run each. No boot-to-boot spread. The speed method has a CV per result; the quality table has margins.
- Sampled cells differ in generated text between engines.
- On the Spark, the fixed-method speed runs were separate server starts from the probe cells.
- The fixed-method speed requests go through a second local recording proxy of my harness (random local port) before they reach the engine, the same hop for all three setups. With TandemLLM that means two hops (recording proxy, then the raw proxy). Decode there is (T-1) / (t(T) - t(1)) from the SSE timestamps, with T from `usage.completion_tokens`. The speed tool was built and proven on llama.cpp; it is not separately validated for ExLlamaV3, TandemLLM or TensorFold.
- All runs are from 19 October 2026.

## Suggested placement (your call)
- Receipts under `docs/receipts/` in the three folders below.
- If you want a doc sentence, one option under "Public probe": "On a clock-capped (2200 MHz) Spark, the public C1 cell with default sampling, with net decode derived from the receipt as sum of completion tokens divided by sum of (wall - TTFT), lands within 1 % of the historical C1 net decode (60.16 against 60.61 tok/s); the C1 aggregate on its own is 52.43 tok/s, and the C4 to C16 aggregates sit 6 to 23 % lower."

## Attachments (receipt JSONs, unchanged from the runs; sha256)
`docs/receipts/rtx5090/`
```
a29a7d275fc1bca4f15edff199193c9b48559ccba0ff97c8ccef4b102894b855  smoke-ready.json
8ece68d28649c16a863ebadb517fb6dcd87cc0f721f96bf531898116959b9c4e  c1-sampled.json
b08329bbef2fe4fb895a6a54af1498af52d6af83ef38cf3f822a482878c1edce  c1-greedy.json
```
`docs/receipts/dgx-spark-tandemllm/`
```
7df2615ce83bee6bc8c1b5362226459db4e497dcaeaca3c2dad9475509b99cd5  sampled-c1-tandem.json
```
`docs/receipts/dgx-spark-tensorfold/`
```
594eb345caa0208e09bd6064093bfe97a4f201a221f10efbf290af366fadbe1c  smoke-ready.json
dc81ae2c3bf652c12d887768d3a7f4b8111bdf6d3e0426d265a7ac2a45f84caa  smoke-thinking-c4.json
deafbe42588cee09a195e953c4719bd90b282a4c5bf225b132db7d684ac1ae6d  public-c1.json
c16022d908c56d8e04a697cfb76a15de3e31fbee9ade6656ef4a2876697ea608  public-c4.json
1ea495696f6cb6559b10a93d7cb64908d291c6e807e1380648ca3f1a133ba54b  public-c8.json
27d7dfd6351744fe86c3916bb4ae1acb8c008cdede5808b0e44c3bc56298faf6  public-c16.json
214cde0f86bc702d21ed1d3f45de7c9e858230af8723ec8cc11327565bad6bb9  sampled-c1.json
```
The receipts hold no host names, paths or power readings.

Credit line: **Benchmarks by Niklas Lenz (@niklaslenz_ai), niklaslenz.com**

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
