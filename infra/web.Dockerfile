# syntax=docker/dockerfile:1.7
FROM node:24.12.0-bookworm-slim AS builder
WORKDIR /workspace/apps/web
ENV CI=true

COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY packages/contracts/ /workspace/packages/contracts/
COPY apps/web/ ./

ARG VITE_API_BASE_URL=
ARG VITE_MAP_STYLE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL} \
    VITE_MAP_STYLE_URL=${VITE_MAP_STYLE_URL}
RUN npm run check:api && npm run build

FROM nginxinc/nginx-unprivileged:1.27.4-alpine3.21 AS runtime
COPY --chown=101:101 infra/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder --chown=101:101 /workspace/apps/web/dist /usr/share/nginx/html

USER 101
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=5 \
  CMD ["wget", "--quiet", "--tries=1", "--spider", "http://127.0.0.1:8080/healthz"]
