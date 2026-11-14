# pathoinfer: the slide job API (and the CLI). One Dockerfile for CPU and CUDA images:
#   docker build -f deploy/Dockerfile -t pathoinfer:cpu .
#   docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=cu130 -t pathoinfer:cuda .
# The CUDA image needs the NVIDIA container runtime and a driver for CUDA 13.0.
# Build from the repository root (the context is filtered by .dockerignore).
FROM python:3.12-slim

ARG TORCH_INDEX=cpu
# The patchology commit pathoinfer is tested against (as in .github/workflows/ci.yml).
ARG PATCHOLOGY_REF=593b13b8bbb197c0cbaf4d806a51fa7376dbe53b

# UV_HTTP_TIMEOUT: the CUDA wheels are large and NVIDIA's index can be slow.
ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=300

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /usr/local/bin/uv

RUN uv pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
RUN uv pip install "patchology @ git+https://github.com/twpkevin06222/patchology@${PATCHOLOGY_REF}"

WORKDIR /opt/pathoinfer
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# --no-sources: ignore the uv workspace's sources (CUDA torch index, local patchology).
RUN uv pip install --no-sources ".[onnx,serve,triton]"

# The demo: a model that marks pixels darker than glass, its client script, a synthetic
# slide and a Triton repository for the model (copied into the demo Triton image).
COPY tests/fixtures/tiny_dark_onnx ./demo/model
COPY deploy/demo ./demo/scripts
RUN python demo/scripts/setup.py demo/data demo/model_repository demo/model

RUN useradd --create-home --uid 1000 pathoinfer \
    && mkdir -p /data /models \
    && chown pathoinfer /data /models
USER pathoinfer
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["pathoinfer", "serve", "api", "--jobs-dir", "/data/jobs", "--port", "8080"]
