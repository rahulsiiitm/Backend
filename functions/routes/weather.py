from flask import Blueprint, request, jsonify
from services.weather_service import get_weather_data

weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({"error": "Missing coordinates"}), 400
    data = get_weather_data(lat, lon)
    if "error" in data:
        return jsonify(data), 500
    return jsonify(data)
