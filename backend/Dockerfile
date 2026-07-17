# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS dependencies

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY backend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install --requirement /tmp/requirements.txt


FROM dependencies AS test

COPY backend/requirements-dev.txt /tmp/requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --requirement /tmp/requirements-dev.txt

WORKDIR /app
COPY . /app

CMD ["pytest", "backend/tests"]


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_DIR=/app/logs

RUN groupadd --gid 10001 slothbearflow \
    && useradd --uid 10001 --gid slothbearflow --create-home --shell /usr/sbin/nologin slothbearflow

COPY --from=dependencies /opt/venv /opt/venv

WORKDIR /app
COPY backend/src /app/backend/src
COPY backend/config /app/backend/config
COPY backend/migrations /app/backend/migrations
COPY backend/scripts /app/backend/scripts
RUN mkdir -p /app/logs /app/data \
    && chown -R slothbearflow:slothbearflow /app/logs /app/data \
    && chmod -R a-w /app/backend

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=8).read()"]

CMD ["uvicorn", "backend.src.slothbearflow_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
