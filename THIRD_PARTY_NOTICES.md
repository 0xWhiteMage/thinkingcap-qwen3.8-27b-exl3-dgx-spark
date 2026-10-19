# Third-party scope and notices

## TandemLLM and independent community receipts

[TandemLLM](https://github.com/0xBakeer/TandemLLM) by 0xBakeer and contributors is the recommended Spark serving engine and is separately [AGPL-3.0](https://github.com/0xBakeer/TandemLLM/blob/f1abbbcf3c4c8e2de6c7d27062ef5b44ca075caf/LICENSE). This repository links to its pinned installer/source/profile; it does not bundle or relicense Tandem engine source, dependencies or weights. Review the upstream obligations for your deployment/distribution. The recipe's MIT licence does not replace those terms.

**Benchmarks by Niklas Lenz** — [GitHub: Nipale-ai](https://github.com/Nipale-ai), [X: @niklaslenz_ai](https://x.com/niklaslenz_ai), [niklaslenz.com](https://niklaslenz.com). His unchanged JSON receipts and original report are under [docs/receipts/](docs/receipts/README.md), sourced from PR #1 head `031ebfda6b840f66dd6728f4b69ced5e5ea9dcc2`. Attribution does not turn contributor observations into maintainer validation; report-only speed/quality summaries remain separate from attached receipts. The quoted report retains its author's provenance.

The RTX 5090 suggestion links to the separately maintained [MiaAI-Lab ExLlamaV3 + MTP kit](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/tree/bbb11db4b7ff688a62841c68e0892fa099bc3362); its source and dependencies are not bundled or relicensed here.

## TensorFold and the vendored patches

TensorFold is Copyright 2026 Ash and contributors, Apache License 2.0. Its retained licence, NOTICE, third-party notices and referenced licence texts are under `third_party/tensorfold/`.

`patches/0001-pr260-rebased.patch` carries BHCC2025 / BHCC Spark Lab's EXL3 CUDA contribution from TensorFold PR #260, rebased onto the pinned Python branch. It includes changes to upstream source, tests and documentation and is **not relicensed under this repository's MIT licence**. `patches/0002-gb10-depth-planning.patch` is a one-line modification to upstream TensorFold source; the resulting TensorFold source remains Apache-2.0. Marked patch filenames and provenance documentation identify the modifications.

## Model, tokenizer and release materials

No target or drafter weights are checked into this Git repository. They are fetched separately at immutable revisions and retain their own licences.

Required Notice: Copyright 2026 BottleCap AI (https://bottlecapai.com)

ThinkingCap target weights and reused release materials retain PolyForm Small Business 1.0.0 plus BottleCap's additional individual personal, non-commercial-use permission. The unchanged [target licence and additional permission](third_party/thinkingcap/LICENSE), [full PolyForm terms](third_party/thinkingcap/LICENSE-PolyForm-Small-Business-1.0.0.txt), [Qwen Apache-2.0 licence](third_party/thinkingcap/LICENSE-Apache-2.0-Qwen.txt), [NOTICE](third_party/thinkingcap/NOTICE) and [attributions](third_party/thinkingcap/ATTRIBUTIONS.md) are included for every recipient. The terms are also available at https://polyformproject.org/licenses/small-business/1.0.0. Qwen upstream Apache-2.0 materials and notices remain separate; they do not make the complete derivative Apache-2.0.

The `z-lab/Qwen3.8-27B-DFlash2` pinned model card states Apache-2.0. Inco AI created DFlash 2; z-lab / Z Lab provides the mirror and original DFlash research/architecture. TensorFold's drafter-linear quantisation setting does not change the downloaded checkpoint files.

The banner, charts, `evidence/source/BENCHMARKS.json` and `RELEASE_PROVENANCE.json` come unchanged from the pinned 0xWhiteMage target release. They remain release materials and are not new benchmark evidence created by this recipe.

ExLlamaV3's format/conversion contribution is credited to Turboderp and contributors; no ExLlamaV3 source or conversion dataset is bundled here. TensorFold's notices describe the separately implemented EXL3 readers and kernels.

## Container and tooling

NVIDIA's base container, CUDA, PyTorch, Triton and other downloaded dependencies keep their separate terms. This repository supplies a Dockerfile, not a redistributed container image. Publishing a future binary image would require its own licence/distribution review.

No contributor endorsement is implied.
