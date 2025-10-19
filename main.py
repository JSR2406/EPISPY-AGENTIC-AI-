# main.py
from fastapi import FastAPI, APIRouter, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timedelta
from enum import Enum
import logging
import random
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Enums for type safety ---
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ModelStatus(str, Enum):
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"
    ERROR = "error"

# --- Pydantic Models ---
class PredictionInput(BaseModel):
    location_id: str = Field(..., example="LOC001")
    cases: int = Field(..., ge=0)
    severity_score: float = Field(..., ge=0.0, le=10.0)
    population_density: float = Field(..., ge=0)
    weather_temp: float

class PredictionOutput(BaseModel):
    location_id: str
    risk_level: RiskLevel
    confidence: float
    risk_score: float
    recommendations: List[str]
    timestamp: str
    model_version: str

# --- ML Model Manager (with fix) ---
class MLModelManager:
    def __init__(self):
        self.model = None
        self.model_status = ModelStatus.NOT_LOADED
    
    def load_model(self):
        # The fix: Ensure the 'models' directory exists before proceeding.
        if not os.path.exists('models'):
            os.makedirs('models')
            logger.info("Created missing 'models' directory.")
        try:
            # We use a dummy model, so no file loading is needed.
            self.model = "DummyModelV2.1"
            self.model_status = ModelStatus.LOADED
            logger.info("Dummy model loaded successfully.")
        except Exception as e:
            self.model_status = ModelStatus.ERROR
            logger.error(f"Critical error during model setup: {e}")

model_manager = MLModelManager()

# --- API Router ---
router = APIRouter()

@router.get("/health", tags=["System"])
async def health_check():
    return {"overall": "healthy" if model_manager.model_status == ModelStatus.LOADED else "degraded"}

@router.post("/model/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_risk(request: PredictionInput):
    if model_manager.model_status != ModelStatus.LOADED:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not available for predictions")
    
    # Dummy prediction logic
    features = [request.cases, request.severity_score]
    if features[0] > 100 or features[1] > 7: prediction = "critical"
    elif features[0] > 50 or features[1] > 5: prediction = "high"
    else: prediction = "medium"
    
    risk_score = min((request.cases / 150.0) * 60 + (request.severity_score / 10.0) * 40, 100.0)
    
    return PredictionOutput(
        location_id=request.location_id, risk_level=prediction,
        confidence=round(random.uniform(0.8, 0.95), 2), risk_score=round(risk_score, 1),
        recommendations=["Increase local testing", "Advise public to follow guidelines"],
        timestamp=datetime.now().isoformat(), model_version="2.1.0"
    )

# --- All Feature Endpoints ---

@router.get("/map/heatmap", tags=["Analytics"])
async def get_heatmap_data():
    locations = [
        {"id": "MUM", "name": "Mumbai", "lat": 19.0760, "lng": 72.8777, "risk": "high", "cases": 145},
        {"id": "DEL", "name": "Delhi", "lat": 28.7041, "lng": 77.1025, "risk": "critical", "cases": 203},
        {"id": "BLR", "name": "Bangalore", "lat": 12.9716, "lng": 77.5946, "risk": "medium", "cases": 87},
    ]
    return {"locations": locations}

@router.get("/analytics/outbreak-probability", tags=["Analytics"])
async def get_outbreak_probability():
    timeline = [{"day": day, "probability": round(min(0.15 + (day * 0.02), 1.0), 3)} for day in range(1, 31)]
    return {"timeline": timeline, "peak_day": timeline[-1], "current_trend": "increasing"}

@router.get("/alerts/active", tags=["Alerts"])
async def get_active_alerts():
    return {"alerts": []} # Return empty for now to prevent notification popups

# --- Mock Endpoints for Dashboard ---
@router.get("/model/info", tags=["System"])
async def get_model_info():
    if model_manager.model_status != ModelStatus.LOADED:
       raise HTTPException(status_code=503, detail="Model is currently unavailable")
    return {"performance_metrics": {"accuracy": 0.92}}

@router.get("/model/health", tags=["System"])
async def get_model_health():
    return {"total_predictions": 450, "average_response_time_ms": 50, "status": "healthy"}

@router.get("/model/metrics", tags=["System"])
async def get_model_metrics():
    return {"predictions_by_risk_level": {"low": 200, "medium": 150, "high": 80, "critical": 20}}

# --- FastAPI App Setup ---
app = FastAPI(title="EpiSPY API v2.1")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting EpiSPY API...")
    model_manager.load_model()