# src/models/orchestrator_agent.py
"""
EpiSPY Orchestrator Agent - Complete Implementation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
import logging
import os
import json
import asyncio
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orchestrator Agent", version="2.0.0")

PERCEPTION_URL = os.getenv("PERCEPTION_URL", "http://perception:8001")
COGNITION_URL = os.getenv("COGNITION_URL", "http://cognition:8002")
REASONING_URL = os.getenv("REASONING_URL", "http://reasoning:8003")

class WorkflowRequest(BaseModel):
    workflow_type: str
    locations: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None

class OrchestratorAgent:
    def __init__(self):
        self.workflows = {}
        self.execution_history = []
    
    async def execute_workflow(self, workflow_type: str, params: Dict) -> Dict:
        workflow_id = f"WF-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"🎯 Starting workflow {workflow_id}: {workflow_type}")
        
        try:
            if workflow_type == "full_analysis":
                result = await self._full_analysis_workflow(params)
            elif workflow_type == "quick_check":
                result = await self._quick_check_workflow(params)
            elif workflow_type == "anomaly_detection":
                result = await self._anomaly_detection_workflow(params)
            elif workflow_type == "emergency_response":
                result = await self._emergency_response_workflow(params)
            else:
                raise ValueError(f"Unknown workflow: {workflow_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_time": execution_time,
                "results": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _full_analysis_workflow(self, params: Dict) -> Dict:
        results = {}
        locations = params.get("locations", ["LOC001", "LOC002", "LOC003"])
        
        # Step 1: Data verification
        logger.info("📊 Step 1: Verifying data")
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.get(f"{PERCEPTION_URL}/data/latest")
        results["data_verification"] = "complete"
        
        # Step 2: Predictions
        logger.info("🔮 Step 2: Generating predictions")
        async with httpx.AsyncClient(timeout=60.0) as client:
            pred_response = await client.post(
                f"{COGNITION_URL}/predict",
                json={"location_codes": locations, "forecast_days": 7}
            )
            predictions = pred_response.json()["predictions"]
        results["predictions"] = predictions
        
        # Step 3: Risk assessment
        logger.info("⚠️ Step 3: Calculating risk")
        risk_assessment = self._calculate_risk(predictions)
        results["risk_assessment"] = risk_assessment
        
        # Step 4: LLM analysis
        logger.info("🤖 Step 4: AI reasoning")
        async with httpx.AsyncClient(timeout=120.0) as client:
            analysis_response = await client.post(
                f"{REASONING_URL}/analyze",
                json={"predictions": predictions}
            )
            analysis = analysis_response.json()
        results["ai_analysis"] = analysis
        
        return results
    
    async def _quick_check_workflow(self, params: Dict) -> Dict:
        locations = params.get("locations", ["LOC001"])
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{COGNITION_URL}/predict",
                json={"location_codes": locations, "forecast_days": 7}
            )
            predictions = response.json()["predictions"]
        
        status_summary = {}
        for pred in predictions:
            status = "⚠️ alert" if pred["trend"] == "increasing" and pred["confidence_score"] > 0.8 else "✅ normal"
            status_summary[pred["location_code"]] = {
                "status": status,
                "trend": pred["trend"],
                "confidence": pred["confidence_score"]
            }
        
        return status_summary
    
    async def _anomaly_detection_workflow(self, params: Dict) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{PERCEPTION_URL}/data/latest", params={"days": 14})
            data = response.json()
        
        from collections import defaultdict
        location_data = defaultdict(list)
        for record in data["records"]:
            location_data[record["location_code"]].append(record["case_count"])
        
        anomalies = []
        for location, cases in location_data.items():
            if len(cases) < 7:
                continue
            mean = np.mean(cases)
            std = np.std(cases)
            for i, value in enumerate(cases[-3:]):
                z_score = (value - mean) / (std + 1e-10)
                if abs(z_score) > 2.5:
                    anomalies.append({
                        "location": location,
                        "value": value,
                        "z_score": float(z_score),
                        "severity": "high" if abs(z_score) > 3 else "moderate"
                    })
        
        return {
            "anomalies_detected": len(anomalies),
            "details": anomalies
        }
    
    async def _emergency_response_workflow(self, params: Dict) -> Dict:
        logger.warning("🚨 EMERGENCY WORKFLOW ACTIVATED")
        locations = params.get("locations", ["LOC001", "LOC002"])
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{COGNITION_URL}/predict",
                json={"location_codes": locations, "forecast_days": 3}
            )
            predictions = response.json()["predictions"]
        
        critical_locations = [
            {
                "location": pred["location_code"],
                "current_cases": pred["current_cases"],
                "urgency": "CRITICAL"
            }
            for pred in predictions
            if pred["trend"] == "increasing" and pred["confidence_score"] > 0.75
        ]
        
        return {
            "status": "EMERGENCY",
            "critical_locations": critical_locations,
            "immediate_actions": [
                "🚨 Activate emergency protocols",
                "📞 Alert health authorities",
                "🏥 Prepare medical resources"
            ]
        }
    
    def _calculate_risk(self, predictions: List[Dict]) -> Dict:
        if not predictions:
            return {"overall_risk": 0, "risk_level": "LOW"}
        
        risk_scores = []
        for pred in predictions:
            current = pred["current_cases"]
            avg_forecast = sum(pred["predicted_cases"]) / len(pred["predicted_cases"])
            growth_rate = (avg_forecast - current) / current if current > 0 else 0
            risk = min(growth_rate * 50 + pred["confidence_score"] * 30, 100)
            risk_scores.append(risk)
        
        overall_risk = sum(risk_scores) / len(risk_scores)
        
        return {
            "overall_risk": overall_risk,
            "risk_level": "CRITICAL" if overall_risk >= 75 else "HIGH" if overall_risk >= 50 else "MODERATE" if overall_risk >= 25 else "LOW"
        }

orchestrator = OrchestratorAgent()

@app.get("/")
async def root():
    return {
        "service": "Orchestrator Agent",
        "version": "2.0.0",
        "workflows": ["full_analysis", "quick_check", "anomaly_detection", "emergency_response"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/workflow/execute")
async def execute_workflow(request: WorkflowRequest):
    try:
        params = request.parameters or {}
        if request.locations:
            params["locations"] = request.locations
        result = await orchestrator.execute_workflow(request.workflow_type, params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/types")
async def get_workflow_types():
    return {
        "workflows": [
            {"type": "full_analysis", "description": "Complete analysis", "time": "60-90s"},
            {"type": "quick_check", "description": "Fast status", "time": "10-15s"},
            {"type": "anomaly_detection", "description": "Detect anomalies", "time": "20-30s"},
            {"type": "emergency_response", "description": "Crisis handling", "time": "15-20s"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)