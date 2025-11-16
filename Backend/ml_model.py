import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

logger = logging.getLogger(__name__)

# Global model variable
model = None

def load_model(model_path='model/disaster_model.pkl'):
    """Load the trained ML model"""
    global model
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("ML model loaded successfully")
        else:
            logger.warning("No trained model found. Using rule-based fallback.")
            model = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

def prepare_features(historical_data):
    """Prepare features for ML prediction from historical data"""
    if len(historical_data) < 5:
        return create_default_features()
    
    try:
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Calculate features for prediction
        features = {}
        
        # Recent values (last reading)
        recent = df.iloc[-1]
        features['temp_current'] = recent['temperature']
        features['hum_current'] = recent['humidity']
        features['gas_current'] = recent['gas_level']
        features['vib_current'] = recent['vibration']
        features['dist_current'] = recent['distance']
        
        # Averages over different time windows
        features['temp_avg_5min'] = df['temperature'].tail(5).mean()
        features['hum_avg_5min'] = df['humidity'].tail(5).mean()
        features['gas_avg_5min'] = df['gas_level'].tail(5).mean()
        
        features['temp_avg_15min'] = df['temperature'].tail(15).mean()
        features['hum_avg_15min'] = df['humidity'].tail(15).mean()
        
        # Trends (slope of linear regression over last 10 points)
        if len(df) >= 10:
            recent_10 = df.tail(10)
            x = np.arange(len(recent_10))
            
            temp_trend = np.polyfit(x, recent_10['temperature'], 1)[0]
            hum_trend = np.polyfit(x, recent_10['humidity'], 1)[0]
            gas_trend = np.polyfit(x, recent_10['gas_level'], 1)[0]
            
            features['temp_trend'] = temp_trend
            features['hum_trend'] = hum_trend
            features['gas_trend'] = gas_trend
        else:
            features['temp_trend'] = 0
            features['hum_trend'] = 0
            features['gas_trend'] = 0
        
        # Maximum values
        features['temp_max_30min'] = df['temperature'].tail(30).max()
        features['gas_max_30min'] = df['gas_level'].tail(30).max()
        
        # Rate of change
        if len(df) >= 2:
            features['temp_change'] = df['temperature'].iloc[-1] - df['temperature'].iloc[-2]
            features['hum_change'] = df['humidity'].iloc[-1] - df['humidity'].iloc[-2]
            features['gas_change'] = df['gas_level'].iloc[-1] - df['gas_level'].iloc[-2]
        else:
            features['temp_change'] = 0
            features['hum_change'] = 0
            features['gas_change'] = 0
        
        return features
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return create_default_features()

def create_default_features():
    """Create default features when insufficient data"""
    return {
        'temp_current': 25, 'hum_current': 50, 'gas_current': 300,
        'vib_current': 0, 'dist_current': 100,
        'temp_avg_5min': 25, 'hum_avg_5min': 50, 'gas_avg_5min': 300,
        'temp_avg_15min': 25, 'hum_avg_15min': 50,
        'temp_trend': 0, 'hum_trend': 0, 'gas_trend': 0,
        'temp_max_30min': 25, 'gas_max_30min': 300,
        'temp_change': 0, 'hum_change': 0, 'gas_change': 0
    }

def predict_disasters(features):
    """Predict disaster risks using ML model or rule-based fallback"""
    
    if model is not None:
        # Use trained ML model
        try:
            # Convert features to array in correct order
            feature_names = [
                'temp_current', 'hum_current', 'gas_current', 'vib_current', 'dist_current',
                'temp_avg_5min', 'hum_avg_5min', 'gas_avg_5min',
                'temp_avg_15min', 'hum_avg_15min',
                'temp_trend', 'hum_trend', 'gas_trend',
                'temp_max_30min', 'gas_max_30min',
                'temp_change', 'hum_change', 'gas_change'
            ]
            
            X = np.array([[features.get(name, 0) for name in feature_names]])
            
            # Get probabilities for each class
            probabilities = model.predict_proba(X)[0]
            
            # Map to disaster types (adjust based on your model classes)
            return {
                'flood': float(probabilities[1] if len(probabilities) > 1 else 0),  # Class 1: Flood
                'landslide': float(probabilities[2] if len(probabilities) > 2 else 0),  # Class 2: Landslide
                'gas_leak': float(probabilities[3] if len(probabilities) > 3 else 0)   # Class 3: Gas Leak
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return rule_based_prediction(features)
    else:
        # Use rule-based prediction as fallback
        return rule_based_prediction(features)

def rule_based_prediction(features):
    """Rule-based prediction when ML model is not available"""
    
    flood_risk = 0
    landslide_risk = 0
    gas_leak_risk = 0
    
    # Flood prediction rules
    if features['dist_current'] < 80:
        flood_risk += 0.3
    if features['dist_current'] < 50:
        flood_risk += 0.4
    if features['hum_current'] > 80:
        flood_risk += 0.2
    if features['hum_trend'] > 0.5:
        flood_risk += 0.1
    
    # Landslide prediction rules  
    if features['vib_current'] == 1:
        landslide_risk += 0.4
    if features['hum_current'] > 85:
        landslide_risk += 0.3
    if features['temp_trend'] < -1:  # Rapid cooling
        landslide_risk += 0.2
    
    # Gas leak prediction rules
    if features['gas_current'] > 600:
        gas_leak_risk += 0.3
    if features['gas_current'] > 800:
        gas_leak_risk += 0.4
    if features['gas_trend'] > 10:  # Rapid gas increase
        gas_leak_risk += 0.3
    
    # Normalize risks to 0-1 scale
    flood_risk = min(1.0, flood_risk)
    landslide_risk = min(1.0, landslide_risk) 
    gas_leak_risk = min(1.0, gas_leak_risk)
    
    return {
        'flood': flood_risk,
        'landslide': landslide_risk,
        'gas_leak': gas_leak_risk
    }