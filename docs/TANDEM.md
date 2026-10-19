# TandemLLM — recommended DGX Spark serving recipe

[README](../README.md) · [Independent recipes and receipts](RECOMMENDED-RECIPES.md) · [Local service inspection](SPARK-TANDEM.md) · [Legacy TensorFold](TENSORFOLD.md)

**Tandem is the primary serving recommendation for this EXL3 release on DGX Spark / ASUS GX10 (GB10).** TensorFold's pinned Python CUDA stack remains available for historical reproduction; it is no longer the default recommendation. This is a recommendation about the serving route, **not a claim that Tandem wins every benchmark**.

The recipe below follows the public community profile in Tandem **v0.12.0**. Niklas Lenz independently measured that version; our existing service was inspected and smoke-tested on **v0.11.4**, not upgraded. A fresh v0.12.0 install/cold start has **not** been repeated by this repository's maintainer. See [verification boundaries](VERIFICATION.md).

## Pinned inputs

| Input | Pin |
| --- | --- |
| Engine | [0xBakeer/TandemLLM v0.12.0](https://github.com/0xBakeer/TandemLLM/tree/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf), commit `f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf` |
| Installer | `install.sh` at that commit; SHA-256 `8907b0ba5786e137be7b2d0628d2141d53f2e55da2ee6b1ddbb1cd0500aaa4f8` |
| Profile | [`community/thinkingcap-27b-kearuga-exl3-4.00`](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/models/qwen3.8-27b/community/thinkingcap-27b-kearuga-exl3-4.00.json) |
| Target | [`0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw`](https://huggingface.co/0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw/tree/11447639033dd80a4c04536b5b37ed7d642802ee), revision `11447639033dd80a4c04536b5b37ed7d642802ee` |
| Drafters | Profile selects `tandem-exl3`: Tandem's K3 EXL3 drafters and K4 draft head; **not** the legacy TensorFold stock drafter |
| Dependency installation | Upstream installer's pinned package list; its dry run reports torch `2.13.0+cu130`, triton `3.7.1`, transformers `5.12.1` |

Machine-readable identity: [manifests/tandem.json](../manifests/tandem.json). This pins the engine, installer, community entry and target revision. Auxiliary models follow the pinned upstream registry/downloader and its HF checksum checks. We do **not** supply a separately frozen full auxiliary-asset manifest or claim a hermetic installation. Retain the plan, resolved asset revisions and checksums for your run.

The historical TensorFold target pin is `d6df32faa6a8109542b2d5e006d2a09312955905`. It is a different repository snapshot. The inspected resident service's two target weight shards match the historical manifest's SHA-256 values; this does not assert that every file across the two HF snapshots is identical.

## Requirements and safety

- Run on **one idle GB10 Spark-class machine**, Linux aarch64, 128 GB unified memory, working NVIDIA driver and CUDA compiler. Our inspected host reports driver `580.173.02`; this is an observation, not a certified minimum.
- Use the installer's hardware/disk/memory checks and the profile download plan. Allow space for target weights, drafters, setup assistant, compiler/cache data and the upstream **60 GB free-disk margin**. The target's approximately 16.35 GB of weight shards is not total runtime memory.
- **One large serving engine at a time.** Do not run this beside a resident TensorFold, SGLang or Tandem instance. A second installation directory does not isolate GPU memory, shared model/cache directories or the box lock.
- Start with localhost binding. Tandem's direct OpenAI routes are unauthenticated by default; use an SSH tunnel or an authenticated, access-restricted proxy for remote clients. Never expose port 8000 directly to the internet.
- The setup path downloads Wren (the setup assistant) as well as installing code/dependencies. `--setup` skips the external lookup corpus; the lookup drafter can still use the conversation. That can change performance versus corpus-enabled runs.

## 1. Fetch and inspect the pinned installer

Run these commands on a **new or deliberately stopped Spark**, not on the NAS. From the recipe repository:

```bash
git rev-parse HEAD                   # keep this in your own run notes
TANDEM_COMMIT=f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf
TANDEM_DIR="$HOME/TandemLLM"
test ! -e "$TANDEM_DIR" || { printf '%s\n' 'Existing install: inspect/back up first; do not overwrite it.'; exit 1; }
mkdir -p results
curl -fL "https://raw.githubusercontent.com/0xBakeer/TandemLLM/$TANDEM_COMMIT/install.sh" \
  -o results/tandem-install.sh
printf '%s\n' '8907b0ba5786e137be7b2d0628d2141d53f2e55da2ee6b1ddbb1cd0500aaa4f8  results/tandem-install.sh' \
  | sha256sum -c -
bash results/tandem-install.sh --help
bash results/tandem-install.sh --dir "$TANDEM_DIR" --ref "$TANDEM_COMMIT" \
  --setup --no-start --yes --host 127.0.0.1 --port 8000 --dry-run
```

Read the dry-run output before proceeding. `--no-start` prevents this installer step from loading a second engine; it does not make installation itself read-only. Neither a dry run nor a successful dependency install proves model load/inference.

## 2. Install, then inspect the community profile

```bash
bash results/tandem-install.sh --dir "$TANDEM_DIR" --ref "$TANDEM_COMMIT" \
  --setup --no-start --yes --host 127.0.0.1 --port 8000

test "$(git -C "$TANDEM_DIR/src" rev-parse HEAD)" = "$TANDEM_COMMIT"
(cd "$TANDEM_DIR/src" && printf '%s\n' \
  '1c5e7016bf75ce164fc8b85934f03e6e9244d28ca9c21fa9489340303c81bd8f  models/qwen3.8-27b/community/thinkingcap-27b-kearuga-exl3-4.00.json' \
  | sha256sum -c -)
bash "$TANDEM_DIR/run/ops/use-profile.sh" --list
bash "$TANDEM_DIR/run/ops/use-profile.sh" \
  community/thinkingcap-27b-kearuga-exl3-4.00 --dry-run
```

The rendered configuration is **`$TANDEM_DIR/run/ops/serve.env`**, not the template under `src/ops/`. Inspect `HOST`, `PORT`, `MAX_LEN`, `DEFAULT_MAX_TOKENS`, `REASONING_EFFORT`, `CONDUCTOR`, `CORPUS` and the launcher before changing them. Do not dump tokens or private configuration into public receipts.

## 3. Apply the profile and verify the served model

This step downloads/checks assets, then stops the old profile if necessary and starts the selected one. It is a deliberate service change, unlike a dry run.

```bash
bash "$TANDEM_DIR/run/ops/use-profile.sh" \
  community/thinkingcap-27b-kearuga-exl3-4.00
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Check the active profile, `weights.name`, target location and `data[].tandem.loaded`, not just HTTP 200. Follow the installer/controller's logs while CUDA extensions compile and graphs initialise. Keep its rollback intact.

**Do not copy the contributor's recovery shortcuts into the normal recipe.** Niklas reported smoke-check failures, a restored `setup-pending` marker and a manual marker workaround, plus lowering `DISK_MARGIN_GB` to 40. These are disclosed deviations, not validated fixes. If your switch fails, inspect its exit status, `src/logs/engine.log`, health/profile identity and the setup marker. Resolve the failure rather than deleting the marker or reducing the disk safety margin to force a start.

## 4. Test the raw model path

The public model ID is normally `TandemLLM`; confirm with `/v1/models`. Run from the recipe repository and choose new output filenames:

```bash
python3 scripts/probe.py --endpoint http://127.0.0.1:8000 --model TandemLLM \
  --tandem-raw --thinking off --greedy --max-tokens 64 \
  --output results/tandem-raw-ready.json
python3 scripts/probe.py --endpoint http://127.0.0.1:8000 --model TandemLLM \
  --tandem-raw --thinking on --requests 2 --concurrency 1 --max-tokens 1024 \
  --output results/tandem-raw-thinking.json
```

Check zero errors, final answers, `finish_reason=stop` and zero cap hits. The thinking-off readiness answer should be `READY`. The probe requests `enable_thinking` and `reasoning_effort=xhigh`; record the engine's effective policy rather than assuming the server default or conductor applies the same effort. Tagged reasoning may be in `content` with the local `reasoning-format=tags` setting; use usage counts and do not publish private reasoning in diagnostic summaries.

`--tandem-raw` adds **`X-Tandem: raw`**. It bypasses conductor routing/injected packages/session-monitor intervention, **not** speculative decoding or all engine caches. On the inspected v0.11.4 service, raw probes had `usage.tandem.parts.injected=0` and no harness; default-path probes had injected package tokens. Niklas used an additional loopback proxy to supply the same header because the original probe lacked this option. Direct new runs remove that hop and are not transport-identical replays.

## 5. Test the default application path separately

Repeat the probe commands **without `--tandem-raw`**, using fresh receipt paths. With conductor enabled, this measures Tandem as an application: routing, packages, effort policy, monitoring and model generation. Do not mix these timings with raw-engine numbers. Both paths were exercised successfully on our already-resident service; neither is a fresh v0.12.0 qualification.

Example raw API request:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Tandem: raw' \
  -d '{"model":"TandemLLM","messages":[{"role":"user","content":"Explain Python context managers with a short example."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":8192,"temperature":1.0,"top_p":0.95,"top_k":20,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"xhigh"}}'
```

Remove the `X-Tandem` header to use the configured default conductor path. `max_tokens` covers reasoning and answer; small caps do not establish task failure without inspecting finish reason.

## Concurrency: a separate, explicitly local adaptation

The public v0.12.0 `ops/start.sh` does **not** forward `QSE_PARALLEL` or `QSE_SIDE_MAX_LEN` to the server. Merely setting those environment values is not enough. The stock recipe above is the **one-slot route**, matching the scope of Niklas's receipt; it is not our adapted four-slot launcher.

Our existing v0.11.4 runtime keeps upstream source clean and adds these arguments in the **rendered launcher**:

```bash
--parallel "${QSE_PARALLEL:-1}" --side-max-len "${QSE_SIDE_MAX_LEN:-65536}"
```

Its configured values are `QSE_PARALLEL=4`, `QSE_SIDE_MAX_LEN=8192`, `MAX_LEN=262144`, `QSE_PARALLEL_FLOOR_GB=15`. Health confirms one 262,144-token slot plus three 8,192-token side slots. Requests too large for a side window must wait for the main slot. `MAX_QUEUE=16` is queue depth, **not 16 decoding slots**. Do not copy the legacy TensorFold `--parallel 16` profile here or imply four full 262K windows fit.

This optional adaptation is documented for analysis, not applied by the quick start. To adopt it, back up the rendered configuration/launcher, inspect the exact version's CLI, stop/restart deliberately and verify the resulting slot count/window sizes and memory admission. See the [sanitised local inspection](SPARK-TANDEM.md).

## Evidence and upstream references

- [Niklas's Tandem C1 receipt](receipts/dgx-spark-tandemllm/sampled-c1-tandem.json) and [full report](receipts/niklas-pr-1-report.md): independent v0.12.0 raw-path measurement, not the default conductor route.
- [Local v0.11.4 inspection and smoke evidence](../evidence/tandem-inspection.json): settings and counts, no private reasoning.
- Upstream at the pinned commit: [setup](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/docs/setup.md), [community weights](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/docs/community-weights.md), [benchmark paths](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/docs/benchmarks.md), [parallel requests](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/docs/parallel-requests.md).

**Credit:** TandemLLM by [0xBakeer and contributors](https://github.com/0xBakeer/TandemLLM), separately licensed [AGPL-3.0](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/LICENSE). **Benchmarks by Niklas Lenz** — [GitHub](https://github.com/Nipale-ai), [X](https://x.com/niklaslenz_ai), [website](https://niklaslenz.com). Model terms remain separate; see [notices](../THIRD_PARTY_NOTICES.md).
