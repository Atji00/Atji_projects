# syntax=docker/dockerfile:1

# --------------------------------------------------
# Stage 1 : installation des dépendances avec uv
# --------------------------------------------------
FROM python:3.12-slim AS builder

# Version de uv figée pour des builds reproductibles
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

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

# Utilisateur non-root (Heroku n'exécute pas les conteneurs en root)
RUN useradd --create-home --uid 1000 appuser

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
COPY src/ ./src/

# ModelInterpreter crée reports/figures/ au démarrage : le dossier doit être
# accessible en écriture quel que soit l'UID attribué par Heroku
RUN mkdir -p src/scoring_model/reports/figures \
    && chmod -R a+rwX src/scoring_model/reports

USER appuser

EXPOSE 8000

# --proxy-headers : derrière le routeur Heroku (TLS terminé en amont), url_for
# doit générer des URLs https, sinon le CSS est bloqué (mixed content)
# SESSION_SECRET_KEY doit être fourni à l'exécution (docker run -e / heroku config:set)
CMD ["sh", "-c", "exec uvicorn main:app --app-dir src --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
