# Breast Cancer Triage Service

A reproducible MLOps project that trains a breast-cancer classification model, serves predictions through FastAPI, packages the service with Docker, and optionally uses Redis to count prediction requests.

> **Educational project only.** This service is not a medical device and must not be used to make clinical decisions.

## Project links

- **GitHub:** https://github.com/danial-zahid/breast-cancer-triage-service
- **Docker Hub:** https://hub.docker.com/r/daniaaaall/breast-cancer-triage-api

## Overview

The project uses the scikit-learn Breast Cancer Wisconsin dataset. It contains 30 numeric features. In the source dataset, target `0` represents malignant and target `1` represents benign. The service treats **malignant** as the positive class.

Main components:

- **DVC:** tracks dataset versions and stores the dataset objects in the configured local DVC remote (`dvc-remote/`).
- **scikit-learn:** trains and evaluates candidate classification pipelines.
- **FastAPI:** exposes prediction, health, model-information, and request-statistics endpoints.
- **Docker:** builds a multi-stage production image.
- **Docker Compose:** runs the API with Redis.
- **Redis:** stores the prediction request counter when configured.

## Repository structure

```text
breast-cancer-triage-service/
├── app/
│   ├── __init__.py
│   └── main.py
├── training/
│   ├── __init__.py
│   ├── prepare_data.py
│   └── train.py
├── data/
│   └── raw/
│       └── breast_cancer.csv.dvc
├── dvc-remote/
├── .dvc/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── Dockerfile
├── Dockerfile.naive
├── .dockerignore
├── docker-compose.yml
├── docker-compose.hub.yml
├── .env.example
├── .gitignore
├── .dvcignore
├── requirements-train.txt
├── requirements-serve.txt
├── requirements-dev.txt
└── README.md
```

## Requirements

- Windows with Docker Desktop (WSL 2 backend), or Linux with Docker Engine.
- Docker Compose v2 (`docker compose`).
- Python 3.11 recommended for the local environment.
- Git.
- DVC installed in the local Python environment.

## 1. Clone the repository

```cmd
git clone https://github.com/danial-zahid/breast-cancer-triage-service.git
cd breast-cancer-triage-service
```

## 2. Create and activate a Python environment (Windows CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-train.txt
pip install -r requirements-serve.txt
```

If the requirements files contain overlapping packages, install the pinned project requirements as listed above and resolve any reported dependency conflict before running the pipeline.

## 3. Restore the dataset with DVC

The CSV is DVC-tracked. Git stores the DVC pointer file, while the dataset content is stored in the configured DVC remote.

```cmd
dvc pull
```

The repository includes the project’s local `dvc-remote/` directory as required for this assignment. If DVC reports that the configured remote path is unavailable, inspect the remote configuration:

```cmd
dvc remote list
dvc config --list
```

Do not add the generated dataset CSV or the DVC cache to Git.

## 4. Run the training pipeline

Run the complete DVC pipeline:

```cmd
dvc repro
```

Run it again to check that DVC skips unchanged stages:

```cmd
dvc repro
```

The pipeline reads the dataset and parameters from the repository, trains the candidate models, selects the model with the best cross-validation malignant-class recall, and writes model artifacts and metrics to the configured output directory.

To inspect recorded metrics:

```cmd
dvc metrics show
```

The model training configuration is stored in `params.yaml`. The fixed random seed used in the project is `42`.

### Recorded model results

The recorded experiment selected Logistic Regression.

| Metric | Logistic Regression |
|---|---:|
| Cross-validation malignant recall | 0.9576 |
| Cross-validation accuracy | 0.9791 |
| Cross-validation ROC-AUC | 0.9942 |
| Test malignant recall | 0.9512 |
| Test accuracy | 0.9722 |
| Test ROC-AUC | 0.9989 |

The Random Forest candidate recorded cross-validation malignant recall of `0.9333`, cross-validation accuracy of `0.9512`, cross-validation ROC-AUC of `0.9867`, test malignant recall of `0.9024`, test accuracy of `0.9630`, and test ROC-AUC of `0.9975`.

## 5. Run the API locally

After the model artifact has been generated, start the API from the repository root:

```cmd
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation:

- http://localhost:8000/docs

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health status |
| POST | `/predict` | Predict malignant/benign class for one record |
| GET | `/model/info` | Model version, metadata, threshold, and metrics |
| GET | `/stats` | Prediction request count |

The `/predict` endpoint expects all 30 numeric feature fields. Missing fields, invalid values, or negative feature values are rejected by request validation.

Example request body can be generated from the installed scikit-learn dataset:

```cmd
python -c "from sklearn.datasets import load_breast_cancer as l; import json; d=l(); print(json.dumps(dict(zip([n.replace(' ','_') for n in d.feature_names], map(float, d.data[0])))))"
```

Copy the generated JSON into the request body in `/docs`.

## 6. Build the production Docker image

Restore the DVC dataset before building because the Dockerfile training stage uses the dataset.

```cmd
dvc pull
docker build -f Dockerfile --build-arg MODEL_VERSION=1.0.0 --build-arg GIT_SHA=unknown -t breast-cancer-triage:1.0.0 .
```

Run the image without Redis:

```cmd
docker run -d --name breast-cancer-api -p 8000:8000 breast-cancer-triage:1.0.0
```

Check the service:

```cmd
docker ps
docker logs breast-cancer-api
```

Open http://localhost:8000/docs to send a prediction.

Stop and remove the standalone container:

```cmd
docker stop breast-cancer-api
docker rm breast-cancer-api
```

## 7. Configure and run Docker Compose

Create your local `.env` from the example:

```cmd
copy .env.example .env
```

Edit `.env` and set your values. Example:

```env
DOCKERHUB_USERNAME=daniaaaall
IMAGE_TAG=1.0.0
HOST_PORT=8000
MODEL_VERSION=1.0.0
GIT_SHA=unknown
MODEL_PATH=artifacts/model.joblib
DECISION_THRESHOLD=0.5
LOG_LEVEL=INFO
REDIS_PORT=6379
```

The real `.env` file contains local configuration and must not be committed to Git. Commit `.env.example` instead.

Start the stack with a fresh build:

```cmd
docker compose up -d --build
```

Check service status and logs:

```cmd
docker compose ps
docker compose logs
```

Open http://localhost:8000/docs and send several predictions. Then check the Redis-backed counter:

```cmd
curl http://localhost:8000/stats
```

Stop the stack but preserve the Redis volume:

```cmd
docker compose down
```

Start it again:

```cmd
docker compose up -d
```

The Redis counter should persist because the named volume is retained. To remove the stack and its stored Redis data:

```cmd
docker compose down -v
```

## 8. Publish the production image to Docker Hub

Log in:

```cmd
docker login -u daniaaaall
```

Tag the production image with both required tags:

```cmd
docker tag breast-cancer-triage:1.0.0 daniaaaall/breast-cancer-triage-api:1.0.0
docker tag daniaaaall/breast-cancer-triage-api:1.0.0 daniaaaall/breast-cancer-triage-api:latest
```

Push both tags:

```cmd
docker push daniaaaall/breast-cancer-triage-api:1.0.0
docker push daniaaaall/breast-cancer-triage-api:latest
```

Verify the tags in the public Docker Hub repository:

https://hub.docker.com/r/daniaaaall/breast-cancer-triage-api/tags

## 9. Run the published image with Docker Compose (clean-room test)

The clean-room Compose file must use the published image and must not contain a `build:` section.

Create a separate empty directory. Place only these two files inside it:

- `docker-compose.hub.yml`
- `.env`

Example `.env`:

```env
DOCKERHUB_USERNAME=daniaaaall
IMAGE_TAG=1.0.0
HOST_PORT=8000
MODEL_PATH=artifacts/model.joblib
MODEL_VERSION=1.0.0
GIT_SHA=unknown
DECISION_THRESHOLD=0.5
LOG_LEVEL=INFO
```

From that directory, pull and start the stack:

```cmd
docker compose -f docker-compose.hub.yml pull
docker compose -f docker-compose.hub.yml up -d
docker compose -f docker-compose.hub.yml ps
```

Test the API at http://localhost:8000/docs and confirm that a prediction succeeds.

Inspect the pulled image digest:

```cmd
docker image inspect daniaaaall/breast-cancer-triage-api:1.0.0 --format "{{index .RepoDigests 0}}"
```

Compare the digest with the digest displayed for the same tag on Docker Hub.

Stop the clean-room stack:

```cmd
docker compose -f docker-compose.hub.yml down
```

## 10. Useful troubleshooting commands

```cmd
docker ps -a
docker images
docker compose ps
docker compose logs api
docker compose logs redis
docker compose down
dvc status
dvc remote list
git status
git log --oneline --all
```

## 11. Reproducibility and security notes

- Use `dvc pull` after cloning and before running the training pipeline or building the production image.
- Keep the dataset CSV and DVC cache out of Git; keep the DVC pointer and the assignment-required `dvc-remote/` folder in the repository.
- Never commit `.env`, passwords, access tokens, or other secrets.
- Keep `.env.example` updated when configuration variables change.
- Use pinned dependency versions and pinned third-party image tags.
- The API should still serve predictions if Redis is unavailable; request counting is best-effort.
- The production container runs as a non-root user and exposes the API on port `8000`.

## 12. Recorded Docker build measurements

The baseline (`Dockerfile.naive`) measurements recorded during the project were:

| Measurement | Baseline image |
|---|---:|
| Cold build time | 7.64 minutes |
| Rebuild after application-code edit | 0.38 minutes |

Record the actual image sizes and the production Dockerfile measurements from your own Docker output in the assignment report. Do not estimate these values.

## License

This repository was created for an academic MLOps assignment.
