# Source, patch and editorial provenance

## Immutable model inputs

| Role | Repository | Revision |
| --- | --- | --- |
| Target | `0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw` | `d6df32faa6a8109542b2d5e006d2a09312955905` |
| Direct BF16 source of the quantisation | `bottlecapai/ThinkingCap-Qwen3.8-27B` | `4bd4e11054e4ceb0dcecfff2f0d2ffa906f37232` |
| Stock drafter | `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

The target/drafter manifests were generated from HF file metadata at those immutable revisions. They preserve every path and size plus the LFS SHA-256 or ordinary Git blob SHA-1. The target download additionally passed its included `SHA256SUMS` during the preceding mirror operation. No model file is changed by this recipe.

## Runtime reconstruction

Base: upstream `ashhart/TensorFold`, Python CUDA commit `ed78d6fc204d89d90b045bf033d6551e7714f3a1`. Current upstream native 1.x is deliberately not substituted.

`patches/0001-pr260-rebased.patch` is the exact 16-file diff from that base to the source used to build the qualified runtime, local rebased commit `a97861fb7fff56670fe098cd8f5fbdba04099558`. It reconstructs source tree `0c59cc48df1ad49957ce7e56d039f205d905f7f3`; the Docker build checks that tree before the GB10 change. The local rebased commit is **provenance, not a public remote fetch dependency**: the public base plus vendored diff is sufficient to rebuild it.

The contribution is [BHCC2025 / BHCC Spark Lab's PR #260](https://github.com/ashhart/TensorFold/pull/260), final public head `f956850e4a4c701daf6ab91fbce139f4e63d46f0`. The PR is closed and unmerged. Rebased and original commit hashes differ; do not describe it as merged upstream or Kearuga-authored kernels. Read the patch to see its scope: EXL3 linear kernels/prefill, model capacity and multi-stream paths, plus documentation/tests.

`patches/0002-gb10-depth-planning.patch` changes only `DEPTH_CHIPS` to admit SM 12.1 to the existing planner, alongside SM 12.0. It is the same one-line source change used in the resident depth121 build. It does not replace TensorFold's planner with a new Kearuga planner.

The base is the multi-architecture image index for NVIDIA PyTorch 26.07, pinned by digest. Its ARM64 child is the same fixed manifest used by the original Spark build; its AMD64 child is pinned in the runtime manifest too. Runtime Python dependency versions were read from the resident environment, including their active dependency closure. PyTorch and Triton stay in the NVIDIA base; the build checks their observed versions instead of installing different wheels. The resulting Docker image can have a different image ID because build metadata, source line endings and installation paths differ. An AMD64 build is not a 24 GB inference qualification.

## Existing recipes used as references

| Reference | Reviewed revision | What was carried over |
| --- | --- | --- |
| [`qwen3.8-27b-kearuga-sglang-dgx-spark-dflash2`](https://github.com/0xWhiteMage/qwen3.8-27b-kearuga-sglang-dgx-spark-dflash2/tree/8de4b12c7992db5d8093903aa06e94b1c3276698) | `8de4b12c7992db5d8093903aa06e94b1c3276698` | Banner/badge/section rhythm, measured tables, explicit pins, quick-start commands, API use, limitations and credits |
| [`Minimax-H3_ComfyUI-SageAttention-Spectrum`](https://github.com/0xWhiteMage/Minimax-H3_ComfyUI-SageAttention-Spectrum/tree/ba9cb64db92a5da8bac48ff66af6f3068b400ab3) | `ba9cb64db92a5da8bac48ff66af6f3068b400ab3` | Small inspectable assets, dependency checklist and file map |
| [`0xWhiteMage/TensorFold`](https://github.com/0xWhiteMage/TensorFold) | Python base above; native main was inspected separately | Runtime-family boundary and upstream documentation context |

No speed figure, task score, kernel attribution or model licence is copied from another recipe onto this release. The banner and charts are the existing target-release assets, reused unchanged. Historical benchmark JSON and release provenance are copied unchanged into `evidence/source/`; their original text still describes the model-only distribution, not this later separate recipe.

## External sources

- [Pinned target model card](https://huggingface.co/0xWhiteMage/ThinkingCap-Qwen3.8-27B-Kearuga-EXL3-4.00bpw/blob/d6df32faa6a8109542b2d5e006d2a09312955905/README.md)
- [ThinkingCap BF16 card used for publisher-reported context](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B/blob/7891e03dbda2e9b63a5b9610f20e83ea441f8e2c/README.md)
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B)
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
- [TensorFold](https://github.com/ashhart/TensorFold)
- [Stock drafter card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/50307d4c4cde6860d4eee73e2547cd786fe8e8a4/README.md)
- [Inco AI — DFlash 2](https://inco.ai/blog/dflash2/)
- [Z Lab — DFlash](https://github.com/z-lab/dflash)
