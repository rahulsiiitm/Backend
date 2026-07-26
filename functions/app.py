from flask import Flask, jsonify
from flask_cors import CORS
from config import OPENWEATHER_API_KEY
from datetime import datetime

# Import blueprints
from routes.crops import crops_bp
from routes.chat import chat_bp
from routes.user import user_bp
from routes.analysis import analysis_bp
from routes.weather import weather_bp

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(crops_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(user_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(weather_bp)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Agricultural Advisory System (Refactored)'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
