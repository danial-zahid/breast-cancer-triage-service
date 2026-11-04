# Production Docker dependency cache experiment
import os
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import redis
from redis.exceptions import RedisError

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict


# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)


# Environment configuration
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

DECISION_THRESHOLD = float(
    os.getenv("DECISION_THRESHOLD", "0.5")
)

MODEL_VERSION_ENV = os.getenv("MODEL_VERSION", "")
GIT_SHA = os.getenv("GIT_SHA", "unknown")

REDIS_HOST = os.getenv("REDIS_HOST", "").strip()
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


# All 30 dataset features
FEATURE_NAMES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension"
]


# Request validation
class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_radius: float = Field(ge=0)
    mean_texture: float = Field(ge=0)
    mean_perimeter: float = Field(ge=0)
    mean_area: float = Field(ge=0)
    mean_smoothness: float = Field(ge=0)
    mean_compactness: float = Field(ge=0)
    mean_concavity: float = Field(ge=0)
    mean_concave_points: float = Field(ge=0)
    mean_symmetry: float = Field(ge=0)
    mean_fractal_dimension: float = Field(ge=0)

    radius_error: float = Field(ge=0)
    texture_error: float = Field(ge=0)
    perimeter_error: float = Field(ge=0)
    area_error: float = Field(ge=0)
    smoothness_error: float = Field(ge=0)
    compactness_error: float = Field(ge=0)
    concavity_error: float = Field(ge=0)
    concave_points_error: float = Field(ge=0)
    symmetry_error: float = Field(ge=0)
    fractal_dimension_error: float = Field(ge=0)

    worst_radius: float = Field(ge=0)
    worst_texture: float = Field(ge=0)
    worst_perimeter: float = Field(ge=0)
    worst_area: float = Field(ge=0)
    worst_smoothness: float = Field(ge=0)
    worst_compactness: float = Field(ge=0)
    worst_concavity: float = Field(ge=0)
    worst_concave_points: float = Field(ge=0)
    worst_symmetry: float = Field(ge=0)
    worst_fractal_dimension: float = Field(ge=0)


# Model and Redis client
model = None
model_card = None
redis_client = None
prediction_count = 0

REDIS_COUNTER_KEY = "prediction_count"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_card, redis_client

    model_file = Path(MODEL_PATH)

    if not model_file.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    card_file = model_file.parent / "model_card.json"

    if not card_file.is_file():
        raise RuntimeError(f"Model card not found: {card_file}")

    model = joblib.load(model_file)

    with open(card_file, "r") as file:
        model_card = json.load(file)

    if model_card.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("Model feature names do not match API features")

    if len(FEATURE_NAMES) != 30:
        raise RuntimeError("Expected exactly 30 features")

    logger.info("Model loaded successfully")
    logger.info("Model version: %s", model_card["model_version"])

    # Configure Redis only when REDIS_HOST is provided
    if REDIS_HOST:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1
        )
        logger.info("Redis configured: %s:%s", REDIS_HOST, REDIS_PORT)
    else:
        logger.info("Redis is not configured. Using local counter.")

    yield

    if redis_client is not None:
        redis_client.close()

    logger.info("API shutting down")


app = FastAPI(
    title="Breast Cancer Triage API",
    description="ML prediction service for breast cancer classification",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    global prediction_count

    input_data = data.model_dump()

    input_df = pd.DataFrame(
        [[input_data[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    probabilities = model.predict_proba(input_df)[0]

    malignant_class_index = list(model.classes_).index(1)

    malignant_probability = float(
        probabilities[malignant_class_index]
    )

    label = (
        "malignant"
        if malignant_probability >= DECISION_THRESHOLD
        else "benign"
    )

    # Local counter is maintained as a fallback
    prediction_count += 1

    # Increment Redis counter when Redis is configured
    if redis_client is not None:
        try:
            current_count = redis_client.incr(REDIS_COUNTER_KEY)
            logger.info("Redis prediction count: %s", current_count)

        except RedisError as error:
            logger.warning(
                "Redis unavailable. Prediction served without Redis: %s",
                error
            )

    logger.info("Prediction completed: label=%s", label)

    return {
        "label": label,
        "malignant_probability": malignant_probability,
        "threshold": DECISION_THRESHOLD,
        "model_version": model_card["model_version"]
    }


@app.get("/model/info")
def model_info():
    return {
        "model_version": MODEL_VERSION_ENV or model_card["model_version"],
        "model_family": model_card["model_family"],
        "sklearn_version": sklearn.__version__,
        "git_commit": GIT_SHA,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_count": len(FEATURE_NAMES),
        "test_metrics": model_card["metrics"]["test"]
    }


@app.get("/stats")
def stats():
    # Use Redis count when Redis is configured
    if redis_client is not None:
        try:
            count = redis_client.get(REDIS_COUNTER_KEY)

            return {
                "prediction_count": int(count) if count else 0,
                "storage": "redis",
                "redis_available": True
            }

        except RedisError as error:
            logger.warning("Could not read Redis statistics: %s", error)

            return {
                "prediction_count": prediction_count,
                "storage": "local_fallback",
                "redis_available": False
            }

    # Local counter when REDIS_HOST is not configured
    return {
        "prediction_count": prediction_count,
        "storage": "local",
        "redis_available": False
    }