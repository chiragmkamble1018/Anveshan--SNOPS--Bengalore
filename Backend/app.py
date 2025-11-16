from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from collections import deque
import ml_model  # Our custom ML module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'snops_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global data storage
sensor_data = {
    'temperature': 0,
    'humidity': 0, 
    'gas_level': 0,
    'vibration': 0,
    'distance': 0,
    'last_update': None,
    'status': 'NORMAL',
    'cause': 'None'
}

# Store historical data for ML predictions (last 60 minutes)
historical_data = deque(maxlen=60)
alerts_log = []
predictions = {
    'flood_risk': 0,
    'landslide_risk': 0,
    'gas_leak_risk': 0,
    'last_prediction': None
}

# API Configuration
API_KEY = "SNOPS_SECRET_KEY_2024"

prediction_thread = None

@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    """Receive sensor data from NodeMCU via HTTP POST"""
    try:
        data = request.get_json()
        
        # Validate API key
        if data.get('api_key') != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Extract and convert values safely
        temperature = safe_float_convert(data.get('temperature', '0'))
        humidity = safe_float_convert(data.get('humidity', '0'))
        gas_level = safe_int_convert(data.get('gas', '0'))
        distance = safe_int_convert(data.get('distance', '0'))
        
        # Update sensor data
        sensor_data.update({
            'temperature': temperature,
            'humidity': humidity,
            'gas_level': gas_level,
            'distance': distance,
            'status': data.get('status', 'NORMAL'),
            'cause': data.get('cause', 'None'),
            'last_update': datetime.now().strftime('%H:%M:%S')
        })
        
        # Convert status to vibration (for compatibility)
        cause = data.get('cause', '').lower()
        sensor_data['vibration'] = 1 if ('vibration' in cause or 'earthquake' in cause or 'landslide' in cause) else 0
        
        logger.info(f"📡 RECEIVED REAL SENSOR DATA: {sensor_data}")
        
        # Store historical data
        historical_data.append({
            'timestamp': datetime.now(),
            'temperature': sensor_data['temperature'],
            'humidity': sensor_data['humidity'],
            'gas_level': sensor_data['gas_level'],
            'vibration': sensor_data['vibration'],
            'distance': sensor_data['distance']
        })
        
        # Check for alerts
        check_alerts(sensor_data)
        
        # Send to all connected clients
        socketio.emit('sensor_update', sensor_data)
        socketio.emit('historical_update', serialize_historical_data(historical_data))
        
        return jsonify({'status': 'success', 'message': 'Data received'}), 200
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        return jsonify({'error': 'Invalid data format'}), 400

def safe_float_convert(value):
    """Safely convert to float, handling strings with units"""
    try:
        if isinstance(value, str):
            # Remove non-numeric characters except decimal point
            value = ''.join(c for c in value if c.isdigit() or c == '.')
        return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0

def safe_int_convert(value):
    """Safely convert to int"""
    try:
        if isinstance(value, str):
            # Remove non-numeric characters
            value = ''.join(c for c in value if c.isdigit())
        return int(value) if value else 0
    except (ValueError, TypeError):
        return 0

def serialize_historical_data(historical_data):
    """Convert datetime objects to strings for JSON serialization"""
    serialized_data = []
    for item in historical_data:
        serialized_item = item.copy()
        serialized_item['timestamp'] = item['timestamp'].strftime('%H:%M:%S')
        serialized_data.append(serialized_item)
    return serialized_data

def check_alerts(data):
    """Check sensor data against thresholds and trigger alerts"""
    alerts = []
    
    # Use status from NodeMCU or local detection
    if data['status'] == 'ALERT':
        alerts.append((data['cause'].upper(), f"{data['cause']} detected"))
    else:
        # Local fallback detection
        if data['distance'] < 50 and data['distance'] > 0:
            alerts.append(("FLOOD", f"Water level critical: {data['distance']}cm"))
        if data['gas_level'] > 600:
            alerts.append(("GAS LEAK", f"Gas level critical: {data['gas_level']}"))
        if data['vibration'] == 1:
            alerts.append(("LANDSLIDE", "Vibration/Seismic activity detected"))
        if data['temperature'] > 40:
            alerts.append(("FIRE RISK", f"High temperature: {data['temperature']}°C"))
    
    # Log and emit alerts
    for alert_type, message in alerts:
        alert_data = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'sensor_data': data.copy()
        }
        alerts_log.append(alert_data)
        socketio.emit('alert', alert_data)
        logger.warning(f"🚨 ALERT: {alert_type} - {message}")

def run_predictions():
    """Run ML predictions in background thread"""
    global predictions
    
    while True:
        try:
            if len(historical_data) >= 10:  # Need minimum data points
                current_features = ml_model.prepare_features(list(historical_data))
                risk_predictions = ml_model.predict_disasters(current_features)
                
                predictions.update({
                    'flood_risk': risk_predictions.get('flood', 0),
                    'landslide_risk': risk_predictions.get('landslide', 0),
                    'gas_leak_risk': risk_predictions.get('gas_leak', 0),
                    'last_prediction': datetime.now().strftime('%H:%M:%S')
                })
                
                socketio.emit('prediction_update', predictions)
                logger.info(f"🤖 Predictions updated: {predictions}")
            
            time.sleep(300)  # Run predictions every 5 minutes
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            time.sleep(60)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'sensor_data': sensor_data,
        'alerts_count': len(alerts_log)
    })

@socketio.on('connect')
def handle_connect():
    """Send current data when client connects"""
    socketio.emit('sensor_update', sensor_data)
    socketio.emit('prediction_update', predictions)
    socketio.emit('alerts_history', alerts_log[-10:])  # Last 10 alerts
    socketio.emit('historical_update', serialize_historical_data(historical_data))

def initialize_system():
    """Initialize system and start threads"""
    global prediction_thread
    
    # Start prediction thread
    prediction_thread = threading.Thread(target=run_predictions, daemon=True)
    prediction_thread.start()
    
    # Load ML model
    ml_model.load_model()
    
    logger.info("🚀 SNOPS Disaster Alert System Started!")
    logger.info("📍 API Endpoint: http://0.0.0.0:5000/api/sensor-data")
    logger.info("📊 Dashboard: http://localhost:5000")
    logger.info("🔴 MOCK DATA DISABLED - Waiting for real sensor data...")
    logger.info("📡 Ready to receive data from NodeMCU receiver!")

if __name__ == '__main__':
    initialize_system()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)