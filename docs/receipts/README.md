# Independent benchmark receipts — Niklas Lenz

**Benchmarks by Niklas Lenz** — [GitHub: Nipale-ai](https://github.com/Nipale-ai) · [X: @niklaslenz_ai](https://x.com/niklaslenz_ai) · [niklaslenz.com](https://niklaslenz.com).

Source: [PR #1](https://github.com/0xWhiteMage/thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda/pull/1), pinned head [`031ebfda6b840f66dd6728f4b69ced5e5ea9dcc2`](https://github.com/0xWhiteMage/thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda/tree/031ebfda6b840f66dd6728f4b69ced5e5ea9dcc2). All 11 JSON files are copied unchanged from that head; their SHA-256 values match the PR's declarations. Runs are dated **19 October 2026** by the author. This is contributor-supplied evidence, not a local rerun or a new quality qualification.

- [Recommended recipes and interpretation](../RECOMMENDED-RECIPES.md)
- [Original contributor report](niklas-pr-1-report.md), preserved as supplied, including setup notes, proxy source, quality summaries and caveats. Its commands and proposed wording are historical suggestions, not additional locally validated instructions.
- [Checksums](SHA256SUMS)

| Receipt | Role |
| --- | --- |
| [dgx-spark-tandemllm/sampled-c1-tandem.json](dgx-spark-tandemllm/sampled-c1-tandem.json) | Public probe; see settings and limitations |
| [dgx-spark-tensorfold/public-c1.json](dgx-spark-tensorfold/public-c1.json) | Public probe; see settings and limitations |
| [dgx-spark-tensorfold/public-c16.json](dgx-spark-tensorfold/public-c16.json) | Public probe; see settings and limitations |
| [dgx-spark-tensorfold/public-c4.json](dgx-spark-tensorfold/public-c4.json) | Public probe; see settings and limitations |
| [dgx-spark-tensorfold/public-c8.json](dgx-spark-tensorfold/public-c8.json) | Public probe; see settings and limitations |
| [dgx-spark-tensorfold/sampled-c1.json](dgx-spark-tensorfold/sampled-c1.json) | Public probe; see settings and limitations |
| [dgx-spark-tensorfold/smoke-ready.json](dgx-spark-tensorfold/smoke-ready.json) | Readiness only |
| [dgx-spark-tensorfold/smoke-thinking-c4.json](dgx-spark-tensorfold/smoke-thinking-c4.json) | Four-request smoke |
| [rtx5090/c1-greedy.json](rtx5090/c1-greedy.json) | Public probe; see settings and limitations |
| [rtx5090/c1-sampled.json](rtx5090/c1-sampled.json) | Public probe; see settings and limitations |
| [rtx5090/smoke-ready.json](rtx5090/smoke-ready.json) | Readiness only |

Verify from this directory:

```bash
sha256sum -c SHA256SUMS
```

Only these JSONs are attached. The contributor's fixed-method speed, quality and Tandem `exl3_fastrun.py` results are **report-only**; their underlying run files are not in the PR. Do not treat them as receipt-verified benchmarks. Raw-path Tandem requests used an extra loopback proxy adding `X-Tandem: raw`; default conductor behaviour was not measured. The updated [public probe](../../scripts/probe.py) can now send this header directly with `--tandem-raw`, so new runs need no proxy and are not transport-identical replays.
