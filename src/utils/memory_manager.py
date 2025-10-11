# src/utils/memory_manager.py
"""
Redis-based Memory System for Tracking
"""

import redis
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            db=0
        )
        logger.info("✅ Memory Manager initialized")
    
    # ========== PREDICTION ACCURACY TRACKING ==========
    
    def store_prediction(self, location_code: str, prediction: Dict):
        """Store prediction for accuracy tracking"""
        key = f"prediction:{location_code}:{datetime.now().strftime('%Y%m%d')}"
        self.redis_client.setex(
            key,
            604800,  # 7 days
            json.dumps(prediction)
        )
    
    def store_actual_outcome(self, location_code: str, date: str, actual_cases: int):
        """Store actual outcome for comparison"""
        key = f"actual:{location_code}:{date}"
        self.redis_client.setex(key, 604800, str(actual_cases))
    
    def calculate_prediction_accuracy(self, location_code: str, days_back: int = 7) -> Dict:
        """Calculate prediction accuracy over time"""
        accuracies = []
        
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            pred_key = f"prediction:{location_code}:{date}"
            actual_key = f"actual:{location_code}:{date}"
            
            pred_data = self.redis_client.get(pred_key)
            actual_data = self.redis_client.get(actual_key)
            
            if pred_data and actual_data:
                prediction = json.loads(pred_data)
                actual = int(actual_data)
                predicted = prediction.get("predicted_cases", [0])[0]
                
                if actual > 0:
                    error_rate = abs(predicted - actual) / actual
                    accuracy = max(0, 1 - error_rate)
                    accuracies.append(accuracy)
        
        return {
            "location_code": location_code,
            "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "samples": len(accuracies),
            "period_days": days_back
        }
    
    # ========== USER PREFERENCES ==========
    
    def store_user_preference(self, user_id: str, preferences: Dict):
        """Store user preferences"""
        key = f"user:preferences:{user_id}"
        self.redis_client.set(key, json.dumps(preferences))
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict]:
        """Retrieve user preferences"""
        key = f"user:preferences:{user_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    # ========== SYSTEM METRICS ==========
    
    def track_api_call(self, endpoint: str, response_time: float):
        """Track API performance"""
        date_key = datetime.now().strftime('%Y%m%d')
        
        # Increment call counter
        self.redis_client.incr(f"metrics:calls:{endpoint}:{date_key}")
        
        # Store response time
        self.redis_client.lpush(
            f"metrics:response_time:{endpoint}:{date_key}",
            response_time
        )
        self.redis_client.ltrim(f"metrics:response_time:{endpoint}:{date_key}", 0, 999)
        
        # Set expiry (30 days)
        self.redis_client.expire(f"metrics:calls:{endpoint}:{date_key}", 2592000)
        self.redis_client.expire(f"metrics:response_time:{endpoint}:{date_key}", 2592000)
    
    def get_api_metrics(self, endpoint: str) -> Dict:
        """Get API metrics"""
        date_key = datetime.now().strftime('%Y%m%d')
        
        calls = int(self.redis_client.get(f"metrics:calls:{endpoint}:{date_key}") or 0)
        response_times = self.redis_client.lrange(
            f"metrics:response_time:{endpoint}:{date_key}", 0, -1
        )
        
        avg_response_time = 0
        if response_times:
            times = [float(t) for t in response_times]
            avg_response_time = sum(times) / len(times)
        
        return {
            "endpoint": endpoint,
            "calls_today": calls,
            "average_response_time": avg_response_time,
            "date": date_key
        }
    
    # ========== MODEL PERFORMANCE ==========
    
    def store_model_performance(self, model_name: str, metrics: Dict):
        """Store model performance metrics"""
        key = f"model:performance:{model_name}"
        data = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.redis_client.setex(key, 86400, json.dumps(data))  # 24 hours
    
    def get_model_performance(self, model_name: str) -> Optional[Dict]:
        """Get model performance metrics"""
        key = f"model:performance:{model_name}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    # ========== CACHE MANAGEMENT ==========
    
    def cache_set(self, key: str, value: Dict, expiry: int = 3600):
        """Set cache with expiry"""
        full_key = f"cache:{key}"
        self.redis_client.setex(full_key, expiry, json.dumps(value))
    
    def cache_get(self, key: str) -> Optional[Dict]:
        """Get from cache"""
        full_key = f"cache:{key}"
        data = self.redis_client.get(full_key)
        return json.loads(data) if data else None
    
    def cache_delete(self, key: str):
        """Delete from cache"""
        full_key = f"cache:{key}"
        self.redis_client.delete(full_key)
    
    # ========== STATISTICS ==========
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            "total_predictions": int(self.redis_client.get("stats:total_predictions") or 0),
            "total_alerts": int(self.redis_client.get("stats:total_alerts") or 0),
            "cache_size": self.redis_client.dbsize(),
            "memory_used": self.redis_client.info()['used_memory_human'],
            "uptime_seconds": self.redis_client.info()['uptime_in_seconds']
        }
    
    def increment_counter(self, counter_name: str):
        """Increment a counter"""
        self.redis_client.incr(f"stats:{counter_name}")
    
    def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        return int(self.redis_client.get(f"stats:{counter_name}") or 0)