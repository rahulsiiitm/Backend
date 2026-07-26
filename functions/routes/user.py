from flask import Blueprint, request, jsonify
from config import db

user_bp = Blueprint('user', __name__)

@user_bp.route('/get_farmer_profile', methods=['GET'])
def get_farmer_profile():
    try:
        user_id = request.args.get('userId')
        if not user_id: return jsonify({'error': 'userId is required'}), 400
        doc = db.collection('users').document(user_id).collection('profile').document('info').get()
        if not doc.exists: return jsonify({'message': 'Not found'}), 404
        return jsonify({'userId': user_id, 'profile': doc.to_dict()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_bp.route('/update_farmer_profile', methods=['POST'])
def update_farmer_profile():
    try:
        data = request.json
        user_id = data.get('userId')
        updates = data.get('updates')
        if not user_id or not updates: return jsonify({'error': 'userId and updates required'}), 400
        ref = db.collection('users').document(user_id).collection('profile').document('info')
        if not ref.get().exists:
            ref.set(updates)
            return jsonify({'message': 'Created new profile.'}), 201
        ref.update(updates)
        return jsonify({'message': 'Profile updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
