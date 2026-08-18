# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ARG POETRY_VERSION=2.1.4

ENV HOME=/home/app \
    PATH=/app/.venv/bin:$PATH \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PREFECT_HOME=/home/app/.prefect \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

COPY config.yaml generator_versions.yaml groundtruth_contract.yaml main.py prefect_pipeline.py ./
COPY config_case ./config_case
COPY prompts ./prompts
COPY src ./src
COPY templates ./templates

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /home/app/.prefect /data \
    && chown -R app:app /home/app /data

USER app
WORKDIR /data

ENTRYPOINT ["/app/.venv/bin/python", "/app/main.py"]
CMD ["--help"]
