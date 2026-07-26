import uuid
import google.generativeai as genai
from flask import Blueprint, request, jsonify
from datetime import datetime
from google.cloud import firestore
from config import db, GEMINI_API_KEY
from utils import validate_user_id, update_user_activity
from services.hf_service import call_hf_model_api

analysis_bp = Blueprint('analysis', __name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@analysis_bp.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        form_data = dict(request.form)
        user_id = form_data.get('user_id') or form_data.get('userId')
        chat_id = form_data.get('chat_id') or form_data.get('chatId')

        if not validate_user_id(user_id):
            return jsonify({'error': 'Valid user_id is required'}), 400

        update_user_activity(user_id)

        try:
            image_file.seek(0)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            image_data = image_file.read()
            image_file.seek(0)
            
            image_part = {
                "mime_type": image_file.content_type,
                "data": image_data
            }
            
            crop_validation_prompt = "Look at this image and respond with only 'crop' if this is an image of a crop/plant/agricultural product, or 'not crop' if it's not. Give only one of these two responses, nothing else."
            
            crop_response = model.generate_content([crop_validation_prompt, image_part])
            crop_result = crop_response.text.strip().lower()
            
            if crop_result != "crop":
                is_new_chat = False
                if chat_id:
                    chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()
                    if not chat_doc.exists:
                        chat_id = None
                        
                if not chat_id:
                    chat_id = str(uuid.uuid4())
                    is_new_chat = True

                user_message = f"[Image Analysis] Uploaded image"
                bot_message = "The uploaded image does not appear to be a crop or plant. Please upload an image of a crop or plant for analysis."
                
                message_data = [
                    {"sender": "user", "message": user_message, "timestamp": datetime.now(), "type": "image"},
                    {"sender": "bot", "message": bot_message, "timestamp": datetime.now(), "type": "error"}
                ]
                
                try:
                    if is_new_chat:
                        db.collection("users").document(user_id).collection("chats").document(chat_id).set({
                            "createdAt": datetime.now(),
                            "lastMessage": bot_message,
                            "updatedAt": datetime.now(),
                            "messages": message_data
                        })
                    else:
                        db.collection("users").document(user_id).collection("chats").document(chat_id).update({
                            "lastMessage": bot_message,
                            "updatedAt": datetime.now(),
                            "messages": firestore.ArrayUnion(message_data)
                        })
                except Exception as e:
                    pass

                return jsonify({
                    'success': False,
                    'error': 'Not a crop image',
                    'message': bot_message,
                    'chatId': chat_id,
                    'userId': user_id,
                    'is_new_chat': is_new_chat
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Crop validation failed: {str(e)}'
            }), 500

        # Valid crop image, proceed to HF model
        model_response = call_hf_model_api(image_file, is_file=True)
        
        if not model_response.get('success'):
            return jsonify({
                'success': False,
                'error': f'Model prediction failed: {model_response.get("error", "Unknown error")}'
            }), 500
        
        predicted_label = model_response.get('disease', 'Unknown disease')

        prompt = f"""
        A plant has been detected with the condition: {predicted_label}.
        Please explain what this condition is, how it affects the plant, and how a farmer can treat or prevent it if it's a disease.
        If it's healthy, provide care tips. Keep it short and clear.
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            gemini_explanation = response.text
        except Exception as e:
            gemini_explanation = f"Detected: {predicted_label}. Please consult with an agricultural expert for detailed analysis and treatment recommendations."

        is_new_chat = False
        if chat_id:
            chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()
            if not chat_doc.exists:
                chat_id = None
                
        if not chat_id:
            chat_id = str(uuid.uuid4())
            is_new_chat = True

        user_message = f"[Image Analysis] Uploaded plant image"
        bot_message = f"Plant Analysis Result: {predicted_label}\n\n{gemini_explanation}"
        
        message_data = [
            {"sender": "user", "message": user_message, "timestamp": datetime.now(), "type": "image"},
            {"sender": "bot", "message": bot_message, "timestamp": datetime.now(), "type": "analysis"}
        ]
        
        try:
            if is_new_chat:
                db.collection("users").document(user_id).collection("chats").document(chat_id).set({
                    "createdAt": datetime.now(),
                    "lastMessage": bot_message,
                    "updatedAt": datetime.now(),
                    "messages": message_data
                })
            else:
                db.collection("users").document(user_id).collection("chats").document(chat_id).update({
                    "lastMessage": bot_message,
                    "updatedAt": datetime.now(),
                    "messages": firestore.ArrayUnion(message_data)
                })
        except Exception as e:
            pass

        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'gemini_explanation': gemini_explanation,
            'chatId': chat_id,
            'userId': user_id,
            'is_new_chat': is_new_chat
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
