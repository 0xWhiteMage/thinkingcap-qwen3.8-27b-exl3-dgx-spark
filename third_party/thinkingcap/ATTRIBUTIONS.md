# Credits and licenses

Required Notice: Copyright 2026 BottleCap AI (https://bottlecapai.com)

This release is a quantized version of BottleCap AI's ThinkingCap-Qwen3.8-27B, based on Alibaba Cloud / Qwen Team's Qwen3.8-27B. It is not a new ThinkingCap fine-tune. No contributor endorsement is implied.

## Model and release

- **BottleCap AI:** ThinkingCap fine-tune, token-efficiency work and direct source weights, pinned to `4bd4e11054e4ceb0dcecfff2f0d2ffa906f37232`. The unchanged license, additional individual personal-use permission and required notice are retained.
- **Alibaba Cloud / Qwen Team:** original Qwen model, architecture, tokenizer and upstream Apache-2.0 materials.
- **Turboderp and ExLlamaV3 contributors:** EXL3 format and conversion tooling. The tooling's software license is separate from the model license; its software is not bundled here.
- **khronnuz:** community model resources used during preparation and evaluation. The direct source of the final weights is BottleCap's BF16 checkpoint, not community quantized weights.
- **0xWhiteMage / Kearuga:** this quantized release, evaluation and serving integration.

## Separately distributed serving software

- **Ash (`ashhart`) and TensorFold contributors:** the serving engine and existing planning, verification and sampling machinery.
- **BHCC2025 / BHCC Spark Lab:** PR #260 kernel contributions used in the measured runtime. These are not Kearuga-authored kernels.
- **Inco AI:** DFlash 2. **z-lab / Z Lab:** its pinned stock mirror and original DFlash research/architecture. Drafter weights are external and not included here; the measurements do not use the separately published Kearuga NVFP4 drafter.
- **NVIDIA, PyTorch and Triton contributors:** hardware and software toolchain used for evaluation and serving; their binaries are not distributed here.

The model retains PolyForm Small Business 1.0.0 plus BottleCap's additional personal-use permission. Qwen's Apache-2.0 materials and notices are retained separately; they do not make the whole derivative Apache-2.0. External software and drafter licenses remain separate. No unadopted research method is represented as shipped code.

## Original DFlash citations

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}

@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
