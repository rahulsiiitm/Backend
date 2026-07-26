import uuid
import google.generativeai as genai
from flask import Blueprint, request, jsonify
from datetime import datetime
from google.cloud import firestore
from config import db, GEMINI_API_KEY
from utils import validate_user_id, update_user_activity

chat_bp = Blueprint('chat', __name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id') or data.get('userId')
        chat_id = data.get('chat_id') or data.get('chatId')
        
        if not validate_user_id(user_id):
            return jsonify({'error': 'Valid user_id is required'}), 400
            
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        update_user_activity(user_id)
        
        history = []
        is_new_chat = False
        
        if chat_id:
            chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()
            if chat_doc.exists:
                messages = chat_doc.to_dict().get('messages', [])
                history = [f"{msg['sender']}: {msg['message']}" for msg in messages[-5:]]
            else:
                chat_id = None
        
        if not chat_id:
            chat_id = str(uuid.uuid4())
            is_new_chat = True

        prompt = f"""
        You are a friendly agricultural medical assistant. Answer health questions naturally, engage with the user but keep the text short and clear.

        Previous conversation: {history}

        User: {message}
        
        Respond helpfully but always remind users to consult doctors for serious concerns.
        """

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        bot_response = response.text
        
        message_data = [
            {"sender": "user", "message": message, "timestamp": datetime.now()},
            {"sender": "bot", "message": bot_response, "timestamp": datetime.now()}
        ]
        
        if is_new_chat:
            db.collection("users").document(user_id).collection("chats").document(chat_id).set({
                "createdAt": datetime.now(),
                "lastMessage": bot_response,
                "updatedAt": datetime.now(),
                "messages": message_data
            })
        else:
            db.collection("users").document(user_id).collection("chats").document(chat_id).update({
                "lastMessage": bot_response,
                "updatedAt": datetime.now(),
                "messages": firestore.ArrayUnion(message_data)
            })
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'chatId': chat_id,
            'userId': user_id,
            'is_new_chat': is_new_chat
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
