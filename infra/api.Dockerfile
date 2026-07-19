# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12.10

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG UV_VERSION=0.11.7
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/atlaslens-venv
WORKDIR /build

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"
COPY services/api/pyproject.toml services/api/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY services/api/ ./
RUN uv sync --frozen --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ENV PATH=/opt/atlaslens-venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 atlaslens \
    && useradd --uid 10001 --gid atlaslens --no-create-home --shell /usr/sbin/nologin atlaslens \
    && install -d -o atlaslens -g atlaslens /app /tmp/atlaslens

WORKDIR /app
COPY --from=builder /opt/atlaslens-venv /opt/atlaslens-venv
COPY --from=builder --chown=atlaslens:atlaslens /build/alembic.ini ./alembic.ini
COPY --from=builder --chown=atlaslens:atlaslens /build/alembic ./alembic

USER atlaslens
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready', timeout=3).read()"]

CMD ["sh", "-c", "alembic -c /app/alembic.ini upgrade head && exec python -m uvicorn atlaslens_api.main:app --host 0.0.0.0 --port 8000"]
