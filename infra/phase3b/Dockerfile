# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc

ARG UV_VERSION=0.11.7
ARG API_LOCK_SHA256=e29a27194a616def630997968d720b8eadabc6ea0ade0b05aa293e2d8321c982
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INDEX=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    UV_LINK_MODE=copy \
    UV_OFFLINE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/atlaslens/venv

# The image build is a future approved operation. Runtime starts offline.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/uv-venv \
    && PIP_NO_INDEX=false /opt/uv-venv/bin/pip install --no-cache-dir "uv==$UV_VERSION" \
    && groupadd --gid 10001 atlaslens \
    && useradd --uid 10001 --gid atlaslens --no-create-home --shell /usr/sbin/nologin atlaslens \
    && install -d -m 0555 -o root -g root /workspace/input \
    && install -d -m 0750 -o atlaslens -g atlaslens /workspace/work /workspace/output /workspace/checkpoints

WORKDIR /opt/atlaslens/api
COPY services/api/pyproject.toml services/api/uv.lock ./
RUN printf '%s  %s\n' "$API_LOCK_SHA256" uv.lock | sha256sum --check - \
    && UV_OFFLINE=false /opt/uv-venv/bin/uv sync --frozen --no-dev --no-install-project

COPY services/api/src ./src
RUN UV_OFFLINE=false /opt/uv-venv/bin/uv sync --frozen --no-dev --no-editable \
    && /opt/atlaslens/venv/bin/python -c "import atlaslens_api"

WORKDIR /opt/atlaslens
COPY config/corpus/source-policy-v1.json config/corpus/manifest-schema-v1.json config/corpus/leakage-policy-v1.json ./config/corpus/
COPY config/cloud/runpod-phase3b-job-v1.json ./config/cloud/runpod-phase3b-job-v1.json
COPY scripts/phase3b/cloud_job.py ./scripts/phase3b/cloud_job.py

USER atlaslens
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/atlaslens/venv/bin/python", "/opt/atlaslens/scripts/phase3b/cloud_job.py"]
CMD ["validate-config", "--config", "/opt/atlaslens/config/cloud/runpod-phase3b-job-v1.json"]
