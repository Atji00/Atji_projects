
.PHONY: docker_build docker_run docker_start docker_stop docker_logs docker_ps \
		docker_shell docker_rm docker_rmi docker_push docker_login docker_clean \
		docker_compose

# Variables
IMAGE_NAME := credit-scoring-modeling
CONTAINER_NAME := credit-scoring-api
TAG := latest
PORT := 8000
ENV_FILE := .env

# Construire l'image Docker
docker_build:
	docker build -f .docker/Dockerfile -t $(IMAGE_NAME):$(TAG) .

# Créer et démarrer un conteneur (SESSION_SECRET_KEY est lu depuis $(ENV_FILE))
docker_run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):8000 --env-file $(ENV_FILE) $(IMAGE_NAME):$(TAG)

# lancer docker compose
docker_compose:
	docker compose -f .docker/compose.yaml up --build -d

# Démarrer un conteneur existant
docker_start:
	docker start $(CONTAINER_NAME)
	docker logs -f $(CONTAINER_NAME)

# Arrêter le conteneur
docker_stop:
	docker stop $(CONTAINER_NAME)

# Lister les conteneurs actifs
docker_ps:
	docker ps

# Ouvrir un terminal dans le conteneur
docker_shell:
	docker exec -it $(CONTAINER_NAME) /bin/sh

# Supprimer le conteneur arrêté
docker_rm:
	docker rm $(CONTAINER_NAME)

# Supprimer l'image locale
docker_rmi:
	docker rmi $(IMAGE_NAME):$(TAG)

# Se connecter à Docker Hub
docker_login:
	docker login

# Publier l'image sur Docker Hub
# Définir DOCKERHUB_USER avec ton identifiant Docker Hub
docker_push:
	docker tag $(IMAGE_NAME):$(TAG) $(DOCKERHUB_USER)/$(IMAGE_NAME):$(TAG)
	docker push $(DOCKERHUB_USER)/$(IMAGE_NAME):$(TAG)

# Nettoyage des ressources inutilisées
docker_clean:
	docker system prune
