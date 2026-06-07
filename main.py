from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

# ============================================================
# Configuration
# ============================================================
APP_NAME = "Intent Classification API"
APP_VERSION = "1.0.0"

MODEL_PATH = Path("SVCmodel.joblib") # chemin vers le modèle sauvegardé
METRICS_PATH = Path("metrics.json")  # ou metrics.csv

# ============================================================
# Initialisation de FastAPI
# ============================================================
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API de classification d'intention, exposée via FastAPI (pipeline sklearn sauvegardé avec joblib)"
)

# Variables globales chargées au démarrage
model: Optional[Pipeline] = None
model_info: Dict[str, Any] = {}
# metrics_data: Dict[str, Any] = {}


# ============================================================
# Schémas Pydantic
# ============================================================

# Structure de vérification (health)
class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    model_loaded: bool

# Structure des Infos du modèle
class ModelInfoResponse(BaseModel):
    model_name: str
    vectorizer: Optional[str] = None
    transformer: Optional[str] = None
    classes: List[str] = []
    # metrics: Dict[str, Any] = {}
    training_info: Dict[str, Any] = {}

# Structure des requêtes
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à classifier")

# Structure des réponse
class ClassifyResponse(BaseModel):
    intent: str
    # confidence: float
    model: str


# ============================================================
# Fonctions utilitaires
# ============================================================

# Chargement du modèle avec JobLib
def load_model(model_path: Path) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable : {model_path}")

    loaded = joblib.load(model_path)

    # on a dump directement le Pipeline
    if isinstance(loaded, Pipeline):
        return loaded

    raise ValueError(
        "Le fichier joblib ne contient pas un Pipeline sklearn."
    )


def extract_model_info(loaded_obj: Any, pipeline_obj: Pipeline) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    # Infos pipeline
    clf = pipeline_obj.named_steps.get("clf")
    vect = pipeline_obj.named_steps.get("vect")
    tfidf = pipeline_obj.named_steps.get("tfidf")

    info["model_name"] = clf.__class__.__name__ if clf else "UnknownModel"
    info["vectorizer"] = vect.__class__.__name__ if vect else None
    info["transformer"] = tfidf.__class__.__name__ if tfidf else None

    return info


# ============================================================
# Événement de démarrage
# ============================================================
@app.on_event("startup") # appelé automatiquement au démarrage de l'api
def startup_event() -> None:
    global model, model_info

    loaded_obj = joblib.load(MODEL_PATH)
    model = load_model(MODEL_PATH)
    model_info = extract_model_info(loaded_obj, model)
    

# ============================================================
# Endpoints
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=APP_NAME,
        version=APP_VERSION,
        model_loaded=model is not None,
    )

@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return ModelInfoResponse(
        model_name=model_info.get("model_name", "UnknownModel"),
        vectorizer=model_info.get("vectorizer"),
        transformer=model_info.get("transformer"),
        classes=model_info.get("classes", []),
        training_info=model_info.get("training_info", {})
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> ClassifyResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le champ 'text' est vide")

    prediction = model.predict([text])[0]
    # confidence = compute_confidence(model, [text])[0]

    return ClassifyResponse(
        intent=str(prediction),
        # confidence=round(confidence, 6),
        model=model_info.get("model_name", "UnknownModel")
    )