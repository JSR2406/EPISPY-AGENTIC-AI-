# src/models/alert_manager.py
"""
Enhanced Alert System with Smart Recommendations
"""

from typing import List, Dict
from datetime import datetime
import redis
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.alert_history_key = "epispy:alert_history"
        self.severity_thresholds = {
            "critical": 75,
            "high": 50,
            "moderate": 25,
            "low": 0
        }
    
    def calculate_severity(self, predictions: List[Dict]) -> str:
        """Calculate alert severity based on predictions"""
        if not predictions:
            return "low"
        
        risk_scores = []
        for pred in predictions:
            current = pred.get("current_cases", 0)
            predicted = pred.get("predicted_cases", [])
            
            if not predicted:
                continue
            
            avg_forecast = sum(predicted) / len(predicted)
            growth_rate = (avg_forecast - current) / current if current > 0 else 0
            confidence = pred.get("confidence_score", 0.5)
            
            # Calculate risk score
            risk = growth_rate * 50 + confidence * 30
            if pred.get("trend") == "increasing":
                risk += 20
            
            risk_scores.append(risk)
        
        if not risk_scores:
            return "low"
        
        avg_risk = sum(risk_scores) / len(risk_scores)
        
        if avg_risk >= self.severity_thresholds["critical"]:
            return "critical"
        elif avg_risk >= self.severity_thresholds["high"]:
            return "high"
        elif avg_risk >= self.severity_thresholds["moderate"]:
            return "moderate"
        else:
            return "low"
    
    def generate_recommendations(self, severity: str, predictions: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on severity"""
        recommendations = []
        
        if severity == "critical":
            recommendations = [
                "🚨 Activate Level 1 Emergency Response immediately",
                "📞 Notify all health authorities and stakeholders",
                "🏥 Mobilize emergency medical teams and resources",
                "📢 Issue public health emergency advisory",
                "🎯 Deploy rapid response teams to high-risk areas",
                "💉 Expedite vaccine distribution and testing kits",
                "🏨 Prepare isolation and quarantine facilities"
            ]
        elif severity == "high":
            recommendations = [
                "⚠️ Increase surveillance in affected areas",
                "📊 Implement daily monitoring and reporting",
                "🏥 Prepare additional hospital capacity",
                "🔬 Conduct targeted testing campaigns",
                "📋 Review and update emergency response plans",
                "👥 Brief emergency response teams"
            ]
        elif severity == "moderate":
            recommendations = [
                "📈 Continue routine monitoring protocols",
                "💉 Ensure adequate vaccine supply",
                "📚 Update public health education materials",
                "🏥 Monitor hospital capacity weekly",
                "📊 Track trend changes closely"
            ]
        else:
            recommendations = [
                "✅ Maintain standard surveillance protocols",
                "📋 Continue regular data collection",
                "📊 Monitor for any trend changes"
            ]
        
        # Add location-specific recommendations
        high_risk_locations = [
            pred["location_code"] 
            for pred in predictions 
            if pred.get("trend") == "increasing" and pred.get("confidence_score", 0) > 0.7
        ]
        
        if high_risk_locations:
            recommendations.insert(1, f"🎯 Focus resources on: {', '.join(high_risk_locations)}")
        
        return recommendations
    
    def create_alert(self, predictions: List[Dict], analysis: Dict) -> Dict:
        """Create comprehensive alert"""
        severity = self.calculate_severity(predictions)
        recommendations = self.generate_recommendations(severity, predictions)
        
        affected_locations = [pred["location_code"] for pred in predictions]
        
        # Generate summary
        if severity == "critical":
            summary = "CRITICAL: Rapid outbreak detected with high confidence. Immediate action required."
        elif severity == "high":
            summary = "HIGH ALERT: Significant increase predicted in multiple locations."
        elif severity == "moderate":
            summary = "MODERATE: Gradual increase observed. Enhanced monitoring recommended."
        else:
            summary = "LOW: Normal patterns observed. Standard protocols sufficient."
        
        alert = {
            "alert_id": f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "affected_locations": affected_locations,
            "summary": summary,
            "detailed_analysis": analysis.get("detailed_analysis", "Analysis pending"),
            "recommendations": recommendations,
            "forecasts": predictions,
            "metadata": {
                "generated_by": "EpiSPY Alert Manager v2.0",
                "confidence_level": "high" if severity in ["critical", "high"] else "moderate"
            }
        }
        
        # Store in history
        self.store_alert(alert)
        
        return alert
    
    def store_alert(self, alert: Dict):
        """Store alert in Redis with history"""
        try:
            alert_json = json.dumps(alert)
            
            # Store in list (keep last 100)
            self.redis_client.lpush(self.alert_history_key, alert_json)
            self.redis_client.ltrim(self.alert_history_key, 0, 99)
            
            # Store by ID with 30-day expiry
            alert_key = f"epispy:alert:{alert['alert_id']}"
            self.redis_client.setex(alert_key, 2592000, alert_json)  # 30 days
            
            # Update statistics
            self.redis_client.incr("stats:total_alerts")
            self.redis_client.incr(f"stats:alerts_{alert['severity']}")
            
            logger.info(f"✅ Alert {alert['alert_id']} stored successfully")
        except Exception as e:
            logger.error(f"❌ Failed to store alert: {e}")
    
    def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve alert history"""
        try:
            alerts = self.redis_client.lrange(self.alert_history_key, 0, limit - 1)
            return [json.loads(alert) for alert in alerts]
        except Exception as e:
            logger.error(f"❌ Failed to retrieve history: {e}")
            return []
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Dict]:
        """Retrieve specific alert"""
        try:
            alert_key = f"epispy:alert:{alert_id}"
            alert_json = self.redis_client.get(alert_key)
            return json.loads(alert_json) if alert_json else None
        except Exception as e:
            logger.error(f"❌ Failed to retrieve alert {alert_id}: {e}")
            return None
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        try:
            return {
                "total_alerts": int(self.redis_client.get("stats:total_alerts") or 0),
                "critical_alerts": int(self.redis_client.get("stats:alerts_critical") or 0),
                "high_alerts": int(self.redis_client.get("stats:alerts_high") or 0),
                "moderate_alerts": int(self.redis_client.get("stats:alerts_moderate") or 0),
                "low_alerts": int(self.redis_client.get("stats:alerts_low") or 0)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {}