# Pipeline CI/CD — Credit Scoring App

Ce dossier contient le workflow GitHub Actions qui teste l'application puis la déploie sur **Heroku** sous forme de conteneur Docker. L'image est aussi publiée sur **Docker Hub**.

Fichier : [`github-docker-heroku-cicd.yaml`](github-docker-heroku-cicd.yaml)

---

## Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Déclencheurs](#déclencheurs)
- [Job 1 : `test` (CI)](#job-1--test-ci)
- [Job 2 : `deploy` (CD)](#job-2--deploy-cd)
- [Configuration requise](#configuration-requise)
- [Images Docker produites](#images-docker-produites)
- [Lancer le pipeline manuellement](#lancer-le-pipeline-manuellement)
- [Dépannage](#dépannage)

---

## Vue d'ensemble

```
push / PR / manuel sur la branche mlops
                │
                ▼
     ┌─────────────────────┐
     │  test (CI)          │  uv sync --locked → pytest
     └─────────┬───────────┘
               │ succès (et pas une pull request)
               ▼
     ┌─────────────────────┐
     │  deploy (CD)        │  build image → Docker Hub → Heroku → release
     └─────────────────────┘
```

| Job      | Rôle                                            | Exécuté sur une PR ? |
|----------|-------------------------------------------------|----------------------|
| `test`   | Installe les dépendances et lance les tests     | Oui                  |
| `deploy` | Construit, publie et déploie l'image Docker     | Non                  |

---

## Déclencheurs

| Événement           | Condition                     |
|---------------------|-------------------------------|
| `push`              | sur la branche `mlops`        |
| `pull_request`      | ciblant la branche `mlops`    |
| `workflow_dispatch` | lancement manuel depuis l'onglet **Actions** |

**Concurrence** : un seul run à la fois par branche (`concurrency` + `cancel-in-progress: true`). Un nouveau push annule le run en cours sur la même branche.

**Permissions** : le `GITHUB_TOKEN` est limité à `contents: read`.

---

## Job 1 : `test` (CI)

Exécuté sur `ubuntu-latest`.

| Étape                 | Description                                                        |
|-----------------------|--------------------------------------------------------------------|
| Checkout repository   | Récupère le code (`actions/checkout@v4`)                           |
| Install Python        | Python 3.12 (`actions/setup-python@v5`)                            |
| Install uv            | Gestionnaire de paquets `uv`, avec cache (`astral-sh/setup-uv@v6`) |
| Install dependencies  | `uv sync --locked` — échoue si `uv.lock` n'est pas à jour avec `pyproject.toml` |
| Run pytest            | `uv run pytest -v` (tests unitaires et d'intégration de `tests/`)  |

Pour reproduire la CI en local :

```bash
uv sync --locked
uv run pytest -v
```

---

## Job 2 : `deploy` (CD)

- Dépend de `test` (`needs: test`) : aucun déploiement si les tests échouent.
- Ignoré pour les pull requests (`if: github.event_name != 'pull_request'`).
- Utilise l'environnement GitHub **`heroku-deployment`**, le seul à exposer les secrets de production.

| Étape                              | Description |
|------------------------------------|-------------|
| Checkout repository                | Récupère le code |
| Install Heroku CLI                 | `npm install --global heroku` |
| Authenticate with Heroku           | `heroku auth:whoami` — vérifie que `HEROKU_API_KEY` est valide |
| Build Docker image                 | Build de [`.docker/Dockerfile`](../../.docker/Dockerfile) en `linux/amd64`, taggé `registry.heroku.com/<app>/web` |
| Login to Docker Hub                | `docker/login-action@v3` |
| Tag image for Docker Hub           | Tags `latest` et SHA court du commit (7 caractères) |
| Push image to Docker Hub           | `docker push --all-tags` |
| Configure Heroku container stack   | `heroku stack:set container` |
| Login to Heroku Container Registry | `heroku container:login` |
| Push image to Heroku               | Push vers `registry.heroku.com/<app>/web` |
| Release application                | `heroku container:release web` — met la nouvelle version en ligne |

> **Note sur le build** : les options `--provenance=false` et `--sbom=false` sont indispensables. Par défaut BuildKit produit une *manifest list* OCI (avec attestations) que le registre Heroku refuse.

---

## Configuration requise

À définir dans **Settings → Environments → `heroku-deployment`** du dépôt GitHub.

### Secrets

| Nom               | Description |
|-------------------|-------------|
| `HEROKU_API_KEY`  | Clé API Heroku (`heroku authorizations:create` ou *Account settings → API Key*) |
| `DOCKERHUB_TOKEN` | Access token Docker Hub (*Account settings → Personal access tokens*), droits *Read & Write* |

### Variables

| Nom                  | Description |
|----------------------|-------------|
| `HEROKU_APP_NAME`    | Nom de l'application Heroku cible |
| `DOCKERHUB_USERNAME` | Nom d'utilisateur Docker Hub |

### Côté Heroku

L'application lit `SESSION_SECRET_KEY` au démarrage. Cette variable n'est pas fournie par le pipeline, il faut la définir une fois sur Heroku :

```bash
heroku config:set SESSION_SECRET_KEY=<valeur-secrète> --app <nom-app>
```

Le port est injecté par Heroku via `$PORT`, repris par la commande `uvicorn` du Dockerfile.

---

## Images Docker produites

| Registre       | Image                                        |
|----------------|----------------------------------------------|
| Heroku         | `registry.heroku.com/<HEROKU_APP_NAME>/web`  |
| Docker Hub     | `<DOCKERHUB_USERNAME>/credit-scoring:latest` |
| Docker Hub     | `<DOCKERHUB_USERNAME>/credit-scoring:<sha7>` |

Le tag `<sha7>` permet de récupérer ou redéployer une version précise :

```bash
docker pull <DOCKERHUB_USERNAME>/credit-scoring:<sha7>
docker run -p 8000:8000 -e SESSION_SECRET_KEY=<valeur> <DOCKERHUB_USERNAME>/credit-scoring:<sha7>
```

---

## Lancer le pipeline manuellement

1. Onglet **Actions** du dépôt GitHub.
2. Sélectionner **Deployment Credit-Scoring App to Heroku**.
3. **Run workflow** → choisir la branche `mlops` → **Run workflow**.

Ou avec la CLI GitHub :

```bash
gh workflow run github-docker-heroku-cicd.yaml --ref mlops
gh run watch
```

---

## Dépannage

| Symptôme | Cause probable / solution |
|----------|---------------------------|
| `uv sync --locked` échoue | `uv.lock` désynchronisé : lancer `uv lock` en local puis commiter le fichier |
| `heroku auth:whoami` échoue | `HEROKU_API_KEY` absent, expiré, ou pas défini dans l'environnement `heroku-deployment` |
| `unsupported` / `manifest invalid` au push Heroku | Vérifier que `--provenance=false --sbom=false` sont bien présents dans le build |
| `denied: requested access to the resource is denied` (Docker Hub) | `DOCKERHUB_TOKEN` sans droit d'écriture ou `DOCKERHUB_USERNAME` erroné |
| Le job `deploy` est *skipped* | Run déclenché par une pull request (comportement voulu) ou job `test` en échec |
| L'app crashe au démarrage sur Heroku | Consulter `heroku logs --tail --app <nom-app>` ; vérifier que `SESSION_SECRET_KEY` est défini |
| CSS bloqué (*mixed content*) | Vérifier que `--proxy-headers` est bien passé à `uvicorn` dans le Dockerfile |
