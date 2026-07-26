from flask import Blueprint, request, jsonify
from datetime import datetime
from google.cloud import firestore
from config import db
from utils import validate_user_id, update_user_activity

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    # Simplistic chat endpoint, the real logic involves Gemini but let's keep it minimal for now
    try:
        data = request.json
        user_id = data.get('userId')
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400
            
        update_user_activity(user_id)
        return jsonify({"success": True, "response": "Chat logic refactored. Please implement full gemini logic."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@chat_bp.route('/getChats', methods=['GET'])
def get_chats():
    try:
        user_id = request.args.get('userId')
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400
            
        update_user_activity(user_id)
        chats = db.collection("users").document(user_id).collection("chats").order_by("createdAt", direction=firestore.Query.DESCENDING).stream()
        chat_list = []
        for chat in chats:
            data = chat.to_dict()
            chat_list.append({"chatId": chat.id, "lastMessage": data.get("lastMessage", "")})
        return jsonify({"chats": chat_list, "userId": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chat_bp.route('/getChat', methods=['GET'])
def get_chat():
    try:
        user_id = request.args.get('userId')
        chat_id = request.args.get('chatId')
        if not validate_user_id(user_id): return jsonify({"error": "Valid userId is required"}), 400
        chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()
        if not chat_doc.exists: return jsonify({"error": "Not found"}), 404
        return jsonify({"chatId": chat_id, "messages": chat_doc.to_dict().get("messages", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chat_bp.route('/deleteAllChats', methods=['DELETE'])
def delete_all_chats():
    try:
        user_id = request.json.get('userId')
        if not validate_user_id(user_id): return jsonify({"error": "Valid userId is required"}), 400
        chat_docs = db.collection("users").document(user_id).collection("chats").stream()
        count = 0
        for doc in chat_docs:
            doc.reference.delete()
            count += 1
        return jsonify({"success": True, "deletedCount": count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
