from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import List, Dict

# Create a router to organize your endpoints
router = APIRouter()

# --- Pydantic Models for Data Validation ---

class LocationsRequest(BaseModel):
    locations: List[str]

class ForecastResponse(BaseModel):
    location_code: str
    current_cases: int
    predicted_cases: int
    trend: str

# --- API Endpoints ---

@router.get("/health")
async def health_check():
    """A simple health check endpoint."""
    return {
        "overall": "healthy",
        "services": {
            "database": "online",
            "ml_model": "ready"
        }
    }

@router.post("/data/ingest")
async def ingest_data():
    """A dummy endpoint for data ingestion."""
    return {"status": "success", "records_processed": 150}

@router.post("/prediction/forecast", response_model=List[ForecastResponse])
async def get_forecast(request: LocationsRequest):
    """A dummy endpoint that returns a 7-day forecast."""
    # Create fake forecast data for the locations in the request
    response_data = []
    for loc in request.locations:
        response_data.append(
            ForecastResponse(
                location_code=loc,
                current_cases=120,
                predicted_cases=145,
                trend="increasing"
            )
        )
    return response_data

@router.post("/alerts/generate")
async def generate_alert(request: LocationsRequest):
    """Generates an AI-powered alert summary."""
    return {
        "severity": "High",
        "summary": f"A significant increasing trend was detected in {len(request.locations)} locations.",
        "recommendations": [
            "Increase local testing and monitoring.",
            "Deploy rapid response teams to affected areas.",
            "Advise public to follow health guidelines."
        ]
    }

# --- FastAPI App Setup ---

# Create the main FastAPI application
app = FastAPI(title="EpiSPY Forecast API")

# Include the router in your application
app.include_router(router)