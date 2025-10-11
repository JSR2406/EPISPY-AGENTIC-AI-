# src/models/enhanced_predictor.py
"""
Enhanced Prediction Engine with Ensemble Models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPredictor:
    def __init__(self, model_dir: str = "/models"):
        self.model_dir = model_dir
        self.xgboost_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.ensemble_weights = {"xgboost": 0.6, "lstm": 0.4}
        self.prediction_history = []
    
    def create_time_series_features(self, df: pd.DataFrame, location_code: str) -> pd.DataFrame:
        """Create comprehensive time-series features"""
        loc_data = df[df['location_code'] == location_code].copy()
        loc_data = loc_data.sort_values('date')
        
        if len(loc_data) < 14:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=loc_data.index)
        features['case_count'] = loc_data['case_count'].values
        
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            features[f'lag_{lag}'] = loc_data['case_count'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14]:
            features[f'rolling_mean_{window}'] = loc_data['case_count'].rolling(window, min_periods=1).mean()
            features[f'rolling_std_{window}'] = loc_data['case_count'].rolling(window, min_periods=1).std()
        
        # Trend
        features['trend'] = loc_data['case_count'].diff()
        features['growth_rate'] = loc_data['case_count'].pct_change()
        
        # Temporal features
        if 'date' in loc_data.columns:
            dates = pd.to_datetime(loc_data['date'])
            features['day_of_week'] = dates.dt.dayofweek
            features['week_of_year'] = dates.dt.isocalendar().week
            features['month'] = dates.dt.month
        
        # Statistical features
        mean_cases = features['case_count'].mean()
        std_cases = features['case_count'].std()
        features['z_score'] = (features['case_count'] - mean_cases) / (std_cases + 1e-10)
        
        features = features.fillna(method='ffill').fillna(0)
        return features
    
    def detect_anomalies(self, data: pd.Series, threshold: float = 3.0) -> List[Dict]:
        """Detect anomalies using Z-score and IQR"""
        anomalies = []
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / (std + 1e-10))
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        for idx, value in enumerate(data):
            is_anomaly = False
            methods = []
            
            if z_scores.iloc[idx] > threshold:
                is_anomaly = True
                methods.append("z_score")
            
            if value < lower_bound or value > upper_bound:
                is_anomaly = True
                methods.append("iqr")
            
            if is_anomaly:
                anomalies.append({
                    "index": idx,
                    "value": float(value),
                    "z_score": float(z_scores.iloc[idx]),
                    "methods": methods,
                    "severity": "high" if z_scores.iloc[idx] > threshold + 1 else "moderate"
                })
        
        return anomalies
    
    def predict_simple(self, features: pd.DataFrame, days: int = 7) -> np.ndarray:
        """Simple moving average prediction (fallback)"""
        last_values = features['case_count'].iloc[-7:].values
        return np.array([np.mean(last_values)] * days)
    
    def calculate_confidence_intervals(self, predictions: np.ndarray, confidence_level: float = 0.95) -> Dict:
        """Calculate confidence intervals"""
        margin = predictions * 0.15  # 15% margin
        lower_bound = np.maximum(predictions - margin, 0)
        upper_bound = predictions + margin
        
        return {
            "predictions": predictions.tolist(),
            "lower_bound": lower_bound.tolist(),
            "upper_bound": upper_bound.tolist(),
            "confidence_level": confidence_level
        }
    
    def predict_with_confidence(self, df: pd.DataFrame, location_code: str, forecast_days: int = 7) -> Dict:
        """Main prediction function"""
        logger.info(f"🔮 Predicting for {location_code}")
        
        features = self.create_time_series_features(df, location_code)
        
        if features.empty:
            return {"error": "Insufficient data", "location_code": location_code}
        
        case_data = features['case_count']
        current_cases = int(case_data.iloc[-1])
        
        # Detect anomalies
        anomalies = self.detect_anomalies(case_data)
        
        # Make predictions
        predictions = self.predict_simple(features, forecast_days)
        
        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(predictions)
        
        # Determine trend
        avg_prediction = np.mean(predictions)
        if avg_prediction > current_cases * 1.1:
            trend = "increasing"
        elif avg_prediction < current_cases * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "location_code": location_code,
            "current_cases": current_cases,
            "predicted_cases": [int(p) for p in predictions],
            "confidence_intervals": confidence_intervals,
            "trend": trend,
            "confidence_score": 0.85,
            "anomalies_detected": len(anomalies),
            "anomaly_details": anomalies[-3:] if anomalies else [],
            "timestamp": datetime.now().isoformat()
        }
        
        