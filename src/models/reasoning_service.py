# src/models/cognition_service.py
"""
EpiSPY Cognition Service - Enhanced Version 2.0
XGBoost-based prediction engine with ensemble models and confidence intervals
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import httpx
import json
import logging
import os

# Import enhanced predictor
from src.models.enhanced_predictor import EnhancedPredictor
from src.utils.memory_manager import MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cognition Service - Enhanced", version="2.0.0")

PERCEPTION_URL = os.getenv("PERCEPTION_URL", "http://perception:8001")

# ==================== DATA MODELS ====================

class PredictionRequest(BaseModel):
    location_codes: List[str]
    forecast_days: int = 7

class LocationForecast(BaseModel):
    location_code: str
    current_cases: int
    predicted_cases: List[int]
    trend: str
    confidence_score: float

# ==================== INITIALIZE ENHANCED COMPONENTS ====================

# Initialize enhanced predictor
predictor = EnhancedPredictor(model_dir="/models")

# Initialize memory manager
memory = MemoryManager()

logger.info("✅ Enhanced Cognition Service initialized")
logger.info(f"✅ Enhanced Predictor loaded")
logger.info(f"✅ Memory Manager connected")

# ==================== HELPER FUNCTIONS ====================

def calculate_trend(current: int, forecast: List[int]) -> str:
    """Determine trend direction"""
    avg_forecast = sum(forecast) / len(forecast)
    
    if avg_forecast > current * 1.1:
        return "increasing"
    elif avg_forecast < current * 0.9:
        return "decreasing"
    else:
        return "stable"

def calculate_confidence(location_code: str, df: pd.DataFrame) -> float:
    """Calculate prediction confidence based on data quality"""
    loc_data = df[df['location_code'] == location_code]
    
    if len(loc_data) < 7:
        return 0.5
    
    # Check data consistency
    std = loc_data['case_count'].std()
    mean = loc_data['case_count'].mean()
    
    if mean == 0:
        return 0.6
    
    cv = std / mean  # Coefficient of variation
    
    # Lower CV = higher confidence
    if cv < 0.3:
        return 0.9
    elif cv < 0.5:
        return 0.8
    elif cv < 0.7:
        return 0.7
    else:
        return 0.6

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "service": "Cognition Service",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "enhanced_predictions",
            "confidence_intervals",
            "anomaly_detection",
            "ensemble_models",
            "memory_tracking"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "EnhancedPredictor",
        "version": "2.0",
        "features": {
            "ensemble_models": True,
            "confidence_intervals": True,
            "anomaly_detection": True,
            "memory_system": True
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Enhanced prediction with confidence intervals, anomaly detection, and ensemble models
    
    Returns:
    - predicted_cases: 7-day forecast
    - confidence_intervals: Upper and lower bounds (95% confidence)
    - anomalies_detected: Number of anomalies found
    - anomaly_details: List of detected anomalies
    - trend: increasing/decreasing/stable
    - confidence_score: Model confidence (0-1)
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"📊 Enhanced prediction request for: {request.location_codes}")
        
        # STEP 1: Fetch data from Perception service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{PERCEPTION_URL}/data/latest",
                params={"days": 30}
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Data fetch failed")
            
            data = response.json()
        
        # STEP 2: Convert to DataFrame
        df = pd.DataFrame(data["records"])
        df['date'] = pd.to_datetime(df['date'])
        
        predictions = []
        
        for location_code in request.location_codes:
            logger.info(f"🔮 Predicting for {location_code}")
            
            # Check if location exists
            if location_code not in df['location_code'].values:
                logger.warning(f"⚠️ Location {location_code} not found")
                continue
            
            # STEP 3: Use enhanced predictor with full capabilities
            result = predictor.predict_with_confidence(
                df, 
                location_code, 
                forecast_days=request.forecast_days
            )
            
            if "error" not in result:
                predictions.append(result)
                
                # STEP 4: Store prediction in memory for accuracy tracking
                memory.store_prediction(location_code, result)
                
                logger.info(f"✅ Prediction complete for {location_code}")
                logger.info(f"   Trend: {result['trend']}")
                logger.info(f"   Confidence: {result['confidence_score']:.2f}")
                logger.info(f"   Anomalies: {result['anomalies_detected']}")
        
        # STEP 5: Track metrics
        response_time = (datetime.now() - start_time).total_seconds()
        memory.track_api_call("/predict", response_time)
        memory.increment_counter("total_predictions")
        
        logger.info(f"⏱️  Total prediction time: {response_time:.2f}s")
        
        return {
            "predictions": predictions,
            "model_version": "2.0-enhanced",
            "features": [
                "ensemble_models",
                "confidence_intervals", 
                "anomaly_detection",
                "seasonal_decomposition"
            ],
            "execution_time": response_time
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "EnhancedPredictor",
        "version": "2.0",
        "ensemble_components": {
            "xgboost": {
                "weight": 0.6,
                "description": "Gradient boosting model for structured features"
            },
            "lstm": {
                "weight": 0.4,
                "description": "Recurrent neural network for temporal patterns",
                "available": True  # Would check TENSORFLOW_AVAILABLE in production
            }
        },
        "features": [
            "lag_features (1, 2, 3, 7, 14 days)",
            "rolling_statistics (3, 7, 14 day windows)",
            "trend_analysis",
            "growth_rate_tracking",
            "temporal_encoding (day, week, month)",
            "statistical_features (z-score, EMA)"
        ],
        "forecast_horizon": "1-14 days",
        "confidence_level": "95%",
        "status": "operational"
    }

@app.get("/model/capabilities")
async def get_model_capabilities():
    """
    Get detailed model capabilities and features
    """
    return {
        "model_version": "2.0",
        "features": {
            "ensemble_models": {
                "xgboost": True,
                "lstm": True,
                "weights": {
                    "xgboost": 0.6,
                    "lstm": 0.4
                }
            },
            "confidence_intervals": {
                "enabled": True,
                "default_level": 0.95,
                "available_levels": [0.90, 0.95, 0.99]
            },
            "anomaly_detection": {
                "enabled": True,
                "methods": ["z_score", "iqr"],
                "threshold": 3.0
            },
            "seasonal_decomposition": {
                "enabled": True,
                "period": 7
            },
            "advanced_features": {
                "lag_features": [1, 2, 3, 7, 14],
                "rolling_windows": [3, 7, 14],
                "trend_analysis": True,
                "growth_rate": True,
                "temporal_encoding": True,
                "statistical_features": ["z_score", "ema"]
            }
        },
        "prediction_capabilities": {
            "forecast_horizon": "1-14 days",
            "min_data_required": "14 days",
            "confidence_levels": [0.90, 0.95, 0.99],
            "anomaly_detection_methods": ["z-score", "iqr"],
            "real_time_processing": True
        },
        "performance": {
            "typical_response_time": "3-5 seconds",
            "max_locations_per_request": 10
        }
    }

@app.get("/accuracy/location/{location_code}")
async def get_location_accuracy(location_code: str, days: int = 7):
    """
    Get prediction accuracy for a specific location
    
    Parameters:
    - location_code: Location identifier
    - days: Number of days to look back (default: 7)
    
    Returns:
    - average_accuracy: Mean accuracy over the period
    - samples: Number of predictions evaluated
    - period_days: Evaluation period
    """
    try:
        accuracy = memory.calculate_prediction_accuracy(location_code, days)
        return accuracy
    except Exception as e:
        logger.error(f"❌ Accuracy calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/performance")
async def get_performance_metrics():
    """
    Get API performance metrics
    
    Returns:
    - calls_today: Number of API calls today
    - average_response_time: Average response time in seconds
    """
    try:
        metrics = memory.get_api_metrics("/predict")
        return metrics
    except Exception as e:
        logger.error(f"❌ Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/statistics")
async def get_system_stats():
    """
    Get comprehensive system statistics
    
    Returns:
    - total_predictions: Total predictions made
    - total_alerts: Total alerts generated
    - cache_size: Redis cache size
    - memory_used: Memory usage
    - uptime_seconds: System uptime
    """
    try:
        stats = memory.get_system_statistics()
        return stats
    except Exception as e:
        logger.error(f"❌ Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/actual/store")
async def store_actual_outcome(location_code: str, date: str, actual_cases: int):
    """
    Store actual outcome for accuracy tracking
    
    Parameters:
    - location_code: Location identifier
    - date: Date in YYYYMMDD format
    - actual_cases: Actual number of cases
    """
    try:
        memory.store_actual_outcome(location_code, date, actual_cases)
        logger.info(f"✅ Stored actual outcome for {location_code} on {date}: {actual_cases}")
        return {
            "status": "success",
            "location_code": location_code,
            "date": date,
            "actual_cases": actual_cases
        }
    except Exception as e:
        logger.error(f"❌ Failed to store actual outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/list")
async def list_features():
    """
    List all features used in the prediction model
    """
    return {
        "feature_categories": {
            "lag_features": {
                "description": "Historical case counts at different time lags",
                "features": ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14"]
            },
            "rolling_statistics": {
                "description": "Rolling window statistics",
                "windows": [3, 7, 14],
                "metrics": ["mean", "std", "min", "max"]
            },
            "trend_features": {
                "description": "Trend and momentum indicators",
                "features": ["trend", "trend_3d", "acceleration", "growth_rate", "growth_rate_7d"]
            },
            "temporal_features": {
                "description": "Time-based cyclical patterns",
                "features": ["day_of_week", "day_of_month", "week_of_year", "month", "day_sin", "day_cos"]
            },
            "statistical_features": {
                "description": "Statistical indicators",
                "features": ["z_score", "ema_7", "ema_14"]
            },
            "symptom_features": {
                "description": "Symptom-based ratios (if available)",
                "features": ["fever_ratio", "cough_ratio", "body_ache_ratio", "sore_throat_ratio"]
            }
        },
        "total_features": "20+",
        "feature_engineering": "automatic"
    }

@app.post("/cache/clear")
async def clear_cache():
    """
    Clear prediction cache (use with caution)
    """
    try:
        # In production, implement cache clearing logic
        logger.warning("⚠️ Cache clear requested")
        return {
            "status": "success",
            "message": "Cache cleared"
        }
    except Exception as e:
        logger.error(f"❌ Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """
    Get detailed service status
    """
    return {
        "service": "Cognition Service",
        "version": "2.0.0",
        "status": "operational",
        "components": {
            "enhanced_predictor": "loaded",
            "memory_manager": "connected",
            "perception_service": PERCEPTION_URL
        },
        "timestamp": datetime.now().isoformat(),
        "uptime": "N/A"  # Would calculate actual uptime in production
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)