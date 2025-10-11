# src/integrations/perception_service.py
"""
EpiSPY Perception Service
Handles data ingestion and preprocessing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Perception Service")

# Data storage path
DATA_DIR = "/app/data"
RAW_DATA_PATH = f"{DATA_DIR}/raw/patient_data.csv"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/timeseries_data.json"

# ==================== DATA MODELS ====================

class PatientRecord(BaseModel):
    timestamp: datetime
    location_code: str
    age_group: str
    fever: bool
    cough: bool
    body_ache: bool
    sore_throat: bool

# ==================== DATA GENERATION ====================

def generate_sample_data(days: int = 30, locations: int = 5) -> pd.DataFrame:
    """Generate synthetic patient data for testing"""
    np.random.seed(42)
    
    location_codes = [f"LOC{str(i+1).zfill(3)}" for i in range(locations)]
    age_groups = ["0-18", "19-35", "36-50", "51-65", "65+"]
    
    records = []
    start_date = datetime.now() - timedelta(days=days)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        # Simulate seasonal pattern + random spike
        base_cases = 50 + 30 * np.sin(day * 2 * np.pi / 30)
        daily_multiplier = np.random.uniform(0.8, 1.5)
        
        for location in location_codes:
            # Location-specific cases
            loc_cases = int(base_cases * daily_multiplier * np.random.uniform(0.7, 1.3))
            
            for _ in range(loc_cases):
                record = {
                    "timestamp": current_date,
                    "location_code": location,
                    "age_group": np.random.choice(age_groups),
                    "fever": np.random.random() > 0.3,
                    "cough": np.random.random() > 0.4,
                    "body_ache": np.random.random() > 0.5,
                    "sore_throat": np.random.random() > 0.6
                }
                records.append(record)
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} sample records")
    return df

def process_raw_data(df: pd.DataFrame) -> dict:
    """Process raw data into time-series format"""
    # Convert timestamp to date
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Aggregate by location and date
    aggregated = df.groupby(['location_code', 'date']).agg({
        'fever': 'sum',
        'cough': 'sum',
        'body_ache': 'sum',
        'sore_throat': 'sum'
    }).reset_index()
    
    # Count total cases
    aggregated['case_count'] = df.groupby(['location_code', 'date']).size().values
    
    # Rename columns
    aggregated.columns = [
        'location_code', 'date', 'fever_count', 'cough_count',
        'body_ache_count', 'sore_throat_count', 'case_count'
    ]
    
    # Convert to dict format
    result = {
        "records": aggregated.to_dict('records'),
        "locations": aggregated['location_code'].unique().tolist(),
        "date_range": {
            "start": str(aggregated['date'].min()),
            "end": str(aggregated['date'].max())
        }
    }
    
    # Convert date objects to strings
    for record in result["records"]:
        record['date'] = str(record['date'])
    
    return result

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "service": "Perception Service",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "data_available": os.path.exists(PROCESSED_DATA_PATH)
    }

@app.post("/ingest")
async def ingest_data():
    """
    Ingest and process patient data
    In production, this would read from a real data source
    For demo, we generate synthetic data
    """
    try:
        logger.info("Starting data ingestion")
        
        # Create directories if they don't exist
        os.makedirs(f"{DATA_DIR}/raw", exist_ok=True)
        os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
        
        # Generate sample data
        df = generate_sample_data(days=30, locations=5)
        
        # Save raw data
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Saved raw data to {RAW_DATA_PATH}")
        
        # Process data
        processed = process_raw_data(df)
        
        # Save processed data
        with open(PROCESSED_DATA_PATH, 'w') as f:
            json.dump(processed, f, indent=2)
        logger.info(f"Saved processed data to {PROCESSED_DATA_PATH}")
        
        return {
            "status": "success",
            "records_processed": len(df),
            "locations": processed["locations"],
            "date_range": processed["date_range"]
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/latest")
async def get_latest_data(days: int = 30):
    """
    Retrieve latest processed data
    Used by Cognition service for predictions
    """
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            # Generate data if not exists
            await ingest_data()
        
        with open(PROCESSED_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        # Filter to requested days
        records = data["records"]
        if days < 30:
            records = records[-days * len(data["locations"]):]
        
        return {
            "records": records,
            "locations": data["locations"],
            "count": len(records)
        }
        
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/location/{location_code}")
async def get_location_data(location_code: str, days: int = 30):
    """Get data for specific location"""
    try:
        with open(PROCESSED_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        # Filter by location
        location_records = [
            r for r in data["records"] 
            if r["location_code"] == location_code
        ]
        
        if days < 30:
            location_records = location_records[-days:]
        
        return {
            "location_code": location_code,
            "records": location_records,
            "count": len(location_records)
        }
        
    except Exception as e:
        logger.error(f"Location data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)