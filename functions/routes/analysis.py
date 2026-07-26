from flask import Blueprint, request, jsonify
from services.hf_service import call_hf_model_api

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Endpoint logic for image analysis
    return jsonify({"success": True, "message": "Endpoint reached. Connect frontend properly."})
