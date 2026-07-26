from flask import Blueprint, request, jsonify
from datetime import datetime
from config import db
from utils import validate_user_id, update_user_activity
from services.weather_service import get_weather_data
from services.gemini_service import generate_farming_suggestions_with_gemini, generate_daily_suggestion_with_gemini

crops_bp = Blueprint('crops', __name__)

@crops_bp.route('/addCrop', methods=['POST'])
def add_crop():
    try:
        data = request.json
        user_id = data.get('userId')
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400
        
        update_user_activity(user_id)
        crop_data = {
            'name': data.get('name'),
            'type': data.get('type'),
            'sowedDate': data.get('sowedDate'),
            'area': data.get('area', ''),
            'createdAt': datetime.now()
        }
        
        doc_ref = db.collection("users").document(user_id).collection("crops").document()
        doc_ref.set(crop_data)
        
        crop_data['id'] = doc_ref.id
        crop_data['createdAt'] = crop_data['createdAt'].isoformat()
        return jsonify({"success": True, "message": "Crop added successfully", "crop": crop_data}), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@crops_bp.route('/getCrops', methods=['GET'])
def get_crops():
    try:
        user_id = request.args.get('userId')
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400
            
        update_user_activity(user_id)
        crops = db.collection("users").document(user_id).collection("crops").stream()
        
        crop_list = []
        for crop in crops:
            data = crop.to_dict()
            data['id'] = crop.id
            if 'createdAt' in data and hasattr(data['createdAt'], 'isoformat'):
                data['createdAt'] = data['createdAt'].isoformat()
            crop_list.append(data)
            
        return jsonify({"success": True, "crops": crop_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@crops_bp.route('/deleteCrop', methods=['DELETE'])
def delete_crop():
    try:
        data = request.json
        user_id = data.get('userId')
        crop_id = data.get('cropId')
        
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400
            
        update_user_activity(user_id)
        db.collection("users").document(user_id).collection("crops").document(crop_id).delete()
        return jsonify({"success": True, "message": "Crop deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@crops_bp.route('/getSuggestions', methods=['GET'])
def get_suggestions():
    try:
        user_id = request.args.get("userId")
        lat = request.args.get('lat', 27.1767, type=float)
        lon = request.args.get('lon', 78.0081, type=float)
        
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400
            
        update_user_activity(user_id)
        
        crops_ref = db.collection("users").document(user_id).collection("crops").stream()
        crops = []
        for doc in crops_ref:
            c = doc.to_dict()
            c['id'] = doc.id
            crops.append(c)
            
        if not crops:
            return jsonify({"error": "No crops found."}), 404
            
        weather_data = get_weather_data(lat, lon)
        suggestions = generate_farming_suggestions_with_gemini(crops, weather_data)
        
        keys = ['first', 'second', 'third', 'fourth']
        formatted = {keys[i] if i < 4 else f'suggestion_{i+1}': s for i, s in enumerate(suggestions)}
        
        return jsonify({"success": True, "suggestions": formatted})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@crops_bp.route('/getDailySuggestion', methods=['GET'])
def get_daily_suggestion():
    try:
        lat = request.args.get('lat', 27.1767, type=float)
        lon = request.args.get('lon', 78.0081, type=float)
        weather_data = get_weather_data(lat, lon)
        suggestion = generate_daily_suggestion_with_gemini(weather_data)
        return jsonify({"success": True, "suggestion": suggestion})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
