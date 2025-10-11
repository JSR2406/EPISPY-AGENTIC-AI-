# src/api/main.py
"""
EpiSPY API Gateway - Main FastAPI Application
Orchestrates all microservices
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import httpx
import redis
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="EpiSPY API",
    description="Epidemic Surveillance and Prediction System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# Service URLs
PERCEPTION_URL = os.getenv("PERCEPTION_URL", "http://perception:8001")
COGNITION_URL = os.getenv("COGNITION_URL", "http://cognition:8002")
REASONING_URL = os.getenv("REASONING_URL", "http://reasoning:8003")

# ==================== DATA MODELS ====================

class SeverityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class LocationForecast(BaseModel):
    location_code: str
    current_cases: int
    predicted_cases: List[int] = Field(..., description="7-day forecast")
    trend: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)

class AlertReport(BaseModel):
    alert_id: str
    timestamp: datetime
    severity: SeverityLevel
    affected_locations: List[str]
    summary: str
    detailed_analysis: str
    recommendations: List[str]
    forecasts: List[LocationForecast]

class PredictionRequest(BaseModel):
    locations: Optional[List[str]] = None
    forecast_days: int = Field(default=7, ge=1, le=14)

# ==================== HELPER FUNCTIONS ====================

async def check_service_health(service_name: str, url: str) -> dict:
    """Check health of a service"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                return {"status": "healthy", "response": response.json()}
            else:
                return {"status": "unhealthy", "error": f"Status code {response.status_code}"}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "EpiSPY API Gateway",
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "prediction": "/prediction/forecast",
            "alerts": "/alerts/generate"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for all services"""
    health_status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check Redis
    try:
        redis_client.ping()
        health_status["services"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
    
    # Check other services
    services = {
        "perception": PERCEPTION_URL,
        "cognition": COGNITION_URL,
        "reasoning": REASONING_URL
    }
    
    for name, url in services.items():
        health_status["services"][name] = await check_service_health(name, url)
    
    # Overall health
    all_healthy = all(
        svc.get("status") == "healthy" 
        for svc in health_status["services"].values()
    )
    health_status["overall"] = "healthy" if all_healthy else "degraded"
    
    return health_status

@app.post("/data/ingest")
async def ingest_data(background_tasks: BackgroundTasks):
    """Trigger data ingestion"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{PERCEPTION_URL}/ingest")
            return response.json()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prediction/forecast", response_model=List[LocationForecast])
async def generate_forecast(request: PredictionRequest):
    """
    Generate epidemic forecasts
    
    This endpoint:
    1. Checks cache for recent predictions
    2. Triggers prediction pipeline if needed
    3. Returns structured forecasts
    """
    logger.info(f"Forecast request: {request}")
    
    # Default locations if none provided
    locations = request.locations or ["LOC001", "LOC002", "LOC003"]
    
    # Check cache
    cache_key = f"forecast:{'-'.join(locations)}:{request.forecast_days}"
    cached = redis_client.get(cache_key)
    
    if cached:
        logger.info("Returning cached forecast")
        return json.loads(cached)
    
    # Trigger prediction
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{COGNITION_URL}/predict",
                json={"location_codes": locations, "forecast_days": request.forecast_days}
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Prediction failed")
            
            forecasts = response.json()["predictions"]
            
            # Cache for 1 hour
            redis_client.setex(cache_key, 3600, json.dumps(forecasts))
            
            return forecasts
            
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/generate", response_model=AlertReport)
async def generate_alert():
    """
    Generate natural language alert report
    
    This endpoint:
    1. Fetches latest forecasts
    2. Sends to reasoning service (LLM)
    3. Returns actionable alert
    """
    logger.info("Generating alert report")
    
    try:
        # Get latest forecasts
        forecast_request = PredictionRequest()
        forecasts = await generate_forecast(forecast_request)
        
        # Send to reasoning service
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{REASONING_URL}/analyze",
                json={"predictions": [f.dict() for f in forecasts]}
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Alert generation failed")
            
            alert_data = response.json()
            
            # Create alert report
            alert = AlertReport(
                alert_id=f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                timestamp=datetime.now(),
                severity=alert_data.get("severity", "moderate"),
                affected_locations=alert_data.get("affected_locations", []),
                summary=alert_data.get("summary", ""),
                detailed_analysis=alert_data.get("detailed_analysis", ""),
                recommendations=alert_data.get("recommendations", []),
                forecasts=forecasts
            )
            
            return alert
            
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics/overview")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = {
            "total_predictions": redis_client.get("stats:predictions") or "0",
            "total_alerts": redis_client.get("stats:alerts") or "0",
            "cache_size": redis_client.dbsize(),
            "uptime": "N/A"
        }
        return stats
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)