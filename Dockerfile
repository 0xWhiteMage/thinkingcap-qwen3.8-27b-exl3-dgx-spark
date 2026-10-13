# TensorFold Python CUDA, not the native Zig 1.x runtime.
# Pinned NVIDIA PyTorch 26.07 index selects linux/arm64 or linux/amd64.
FROM nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

LABEL org.opencontainers.image.source="https://github.com/0xWhiteMage/thinkingcap-qwen3.8-27b-exl3-tensorfold-cuda"
LABEL org.opencontainers.image.description="ThinkingCap EXL3 serving recipe: pinned TensorFold Python CUDA + credited PR260 + GB10 planning"

WORKDIR /opt/tensorfold-src
RUN git init . \
 && git remote add origin https://github.com/ashhart/TensorFold.git \
 && git fetch --depth 1 origin ed78d6fc204d89d90b045bf033d6551e7714f3a1 \
 && git checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = ed78d6fc204d89d90b045bf033d6551e7714f3a1
COPY patches/ /opt/recipe-patches/
RUN cd /opt/recipe-patches && sha256sum -c SHA256SUMS \
 && cd /opt/tensorfold-src \
 && git apply --check /opt/recipe-patches/0001-pr260-rebased.patch \
 && git apply --index /opt/recipe-patches/0001-pr260-rebased.patch \
 && test "$(git write-tree)" = 0c59cc48df1ad49957ce7e56d039f205d905f7f3 \
 && git apply --check --unidiff-zero /opt/recipe-patches/0002-gb10-depth-planning.patch \
 && git apply --index --unidiff-zero /opt/recipe-patches/0002-gb10-depth-planning.patch
COPY requirements-runtime.txt /opt/requirements-runtime.txt
RUN python3 -m pip install --no-cache-dir -r /opt/requirements-runtime.txt \
 && python3 -m pip install --no-cache-dir --no-deps --no-build-isolation . \
 && python3 -c "from importlib.metadata import version; assert version('torch') == '2.13.0a0+9186a08b2c.nv26.7.59513937'; assert version('triton') == '3.7.1+gitf797708c.nv26.7'" \
 && python3 -m compileall -q src/tensorfold \
 && tensorfold serve --help
WORKDIR /workspace
