# Text Similarity Service

This repository contains a small example of a text similarity service. The project exposes a REST API backed by a BERT based regression model and a simple web interface to interact with it.

## Features

- **API**: FastAPI application with two endpoints:
  - `GET /` returns a health message.
  - `POST /predict` receives two sentences and returns a similarity score (0‑5).
- **Model**: the model weights and tokenizer are pulled from [Weights & Biases](https://wandb.ai/) using the environment variables below.
- **UI**: Flask web application that consumes the API.
- **Docker**: both services can be run locally or deployed using Docker Compose.
- **Tests**: a minimal test suite for the API is provided under `similarity_service/tests`.

## Local development

1. Clone the repository and install Docker.
2. Provide the environment variables required by the API. You can create a `.env` file or export them in your shell:

```bash
WANDB_API_KEY=<your_wandb_key>
WANDB_PROJECT=<wandb_project>
WANDB_ARTIFACT=<artifact_name:version>
MODEL_NAME=<huggingface_model>
```

3. Start the services with Docker Compose:

```bash
docker compose -f docker-compose.standalone.yml up --build
```

The API will be available on `http://localhost:8000` and the UI on `http://localhost:5000`.

## Deployment on Azure

1. Log in to your Azure Container Registry (ACR) and build the images:

```bash
az acr login --name <your_acr>

docker build -t <your_acr>.azurecr.io/similarity-api:latest -f similarity_service/Dockerfile similarity_service
docker push <your_acr>.azurecr.io/similarity-api:latest

docker build -t <your_acr>.azurecr.io/similarity-ui:latest -f similarity_ui/Dockerfile similarity_ui
docker push <your_acr>.azurecr.io/similarity-ui:latest
```

2. Update `docker-compose.yml` so that the image names point to your registry.
3. Deploy the multi‑container app:

```bash
az webapp create \
  --resource-group <resource_group> \
  --plan <app_service_plan> \
  --name <webapp_name> \
  --multicontainer-config-type compose \
  --multicontainer-config-file docker-compose.yml
```

There is also a GitHub Actions workflow under `.github/workflows/deploy.yml` that performs these steps automatically when pushing to the `main` branch.