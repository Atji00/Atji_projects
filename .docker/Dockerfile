# syntax=docker/dockerfile:1

# --------------------------------------------------
# Stage 1 : installation des dépendances avec uv
# --------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# uv.lock est multiplateforme (requirements.txt contient des paquets Windows : pywin32)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# --------------------------------------------------
# Stage 2 : image d'exécution de l'API
# --------------------------------------------------
FROM python:3.12-slim

# libgomp1 : OpenMP, requis par xgboost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

COPY --from=builder /app/.venv /app/.venv

# Fichiers nécessaires à l'exécution (ProjectPaths résout la racine sur /app)
COPY configs/ ./configs/
COPY data/processed/processed_test/ ./data/processed/processed_test/
COPY data/intermediate/unprocessed_test/ ./data/intermediate/unprocessed_test/
# --chown : ModelInterpreter crée reports/figures/ au démarrage
COPY src/ ./src/

EXPOSE 8000

# SESSION_SECRET_KEY doit être fourni à l'exécution (docker run -e / --env-file)
CMD ["sh", "-c", "uvicorn main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
