# Sentinel — arm64-first image. Explicitly targets linux/arm64 (GB10 /
# DGX OS is Ubuntu-based aarch64); also builds on linux/amd64 for
# development off-target (this build/test session ran on x86_64).
#
# NOTE: the `ml` extra (torch/onnx/onnxscript) pulls large CUDA wheels by
# default via pip's dependency resolution on some platforms. On GB10,
# install a CUDA-enabled torch build matching the on-device CUDA/driver
# version rather than blindly `pip install torch` — check
# https://pytorch.org's aarch64/CUDA install matrix for the current
# GB10-compatible build before relying on this Dockerfile's default.
FROM --platform=$TARGETPLATFORM python:3.11-slim AS base

ARG TARGETPLATFORM
RUN echo "Building for $TARGETPLATFORM" && \
    if [ "$TARGETPLATFORM" != "linux/arm64" ] && [ "$TARGETPLATFORM" != "linux/amd64" ]; then \
        echo "WARNING: untested platform $TARGETPLATFORM — Sentinel targets aarch64 (GB10) primarily, amd64 for dev" ; \
    fi

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY sentinel ./sentinel
COPY config ./config
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["sentinel"]
CMD ["--help"]
