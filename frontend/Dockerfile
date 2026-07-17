# syntax=docker/dockerfile:1.7

FROM node:20-alpine AS build

ARG PNPM_VERSION=9.15.9
ENV CI=1

RUN corepack enable \
    && corepack prepare "pnpm@${PNPM_VERSION}" --activate

WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm,target=/pnpm/store \
    pnpm config set store-dir /pnpm/store \
    && pnpm install --frozen-lockfile

COPY frontend/config ./config
COPY frontend/src ./src
COPY frontend/tsconfig.json ./tsconfig.json
RUN pnpm build


FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

USER 101:101

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["wget", "--quiet", "--tries=1", "--spider", "http://127.0.0.1:8080/healthz"]
