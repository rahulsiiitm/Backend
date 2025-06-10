from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
import uuid
import hashlib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firebase App
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)

# Firestore DB client
db = firestore.client()

# Load the pre-trained plant disease model
disease_model = load_model("models/plant_disease_model.keras") #Model loaded

FRIENDLY_LABELS = {
    'Pepper__bell___Bacterial_spot': 'Bacterial spot on bell pepper',
    'Pepper__bell___healthy': 'Healthy bell pepper',
    'Potato___Early_blight': 'Potato early blight',
    'Potato___healthy': 'Healthy potato',
    'Potato___Late_blight': 'Potato late blight',
    'Tomato___Target_Spot': 'Tomato target spot',
    'Tomato___Tomato_mosaic_virus': 'Tomato mosaic virus',
    'Tomato___Tomato_YellowLeaf_Curl_Virus': 'Tomato yellow leaf curl virus',
    'Tomato___Bacterial_spot': 'Tomato bacterial spot',
    'Tomato___Early_blight': 'Tomato early blight',
    'Tomato___healthy': 'Healthy tomato',
    'Tomato___Late_blight': 'Tomato late blight',
    'Tomato___Leaf_Mold': 'Tomato leaf mold',
    'Tomato___Septoria_leaf_spot': 'Tomato septoria leaf spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite': 'Tomato spider mite infection'
}

# Preprocess image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # match training size
    img_array = image.img_to_array(img)  # shape: (128, 128, 3)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

# Helper function to generate user ID
def generate_user_id(device_info=None):
    """Generate a unique user ID based on device info or random"""
    if device_info:
        # Create hash from device info for consistent ID
        return hashlib.md5(device_info.encode()).hexdigest()[:16]
    else:
        # Generate random user ID
        return str(uuid.uuid4())[:16]

# Helper function to get or create user ID
def get_or_create_user_id(request_data):
    """Extract user_id from request or create new one"""
    user_id = request_data.get('user_id') or request_data.get('userId')
    
    # If no user_id provided or it's invalid (like '0'), create new one
    if not user_id or user_id in ['0', '', 'null', 'undefined']:
        device_info = request_data.get('device_info', '')
        user_id = generate_user_id(device_info)
        
        # Create user document in Firebase if it doesn't exist
        try:
            user_ref = db.collection("users").document(user_id)
            if not user_ref.get().exists:
                user_ref.set({
                    "createdAt": datetime.now(),
                    "lastActive": datetime.now(),
                    "deviceInfo": device_info
                })
        except Exception as e:
            print(f"Warning: Could not create user document: {e}")
    
    return user_id

# Simple in-memory storage (for beginners)
chat_history = {}
user_consultations = {}

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in environment variables")
    print("Please set it using: export GEMINI_API_KEY='your_actual_key'")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
print("✅ Gemini API configured successfully")

# User management endpoints
@app.route('/createUser', methods=['POST'])
def create_user():
    """Create a new user or return existing user ID"""
    try:
        data = request.get_json() or {}
        device_info = data.get('device_info', '')
        
        user_id = generate_user_id(device_info)
        
        # Create user document in Firebase
        user_ref = db.collection("users").document(user_id)
        if not user_ref.get().exists:
            user_ref.set({
                "createdAt": datetime.now(),
                "lastActive": datetime.now(),
                "deviceInfo": device_info
            })
        
        return jsonify({
            'success': True,
            'userId': user_id,
            'message': 'User created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/getUser/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user information"""
    try:
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_doc.to_dict()
        
        # Update last active
        user_ref.update({"lastActive": datetime.now()})
        
        return jsonify({
            'success': True,
            'userId': user_id,
            'userData': user_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Chat endpoint - handles both new and existing chats
@app.route('/chat', methods=['POST'])
def medical_chat():
    """Medical chat with automatic chat management - creates chat only when user sends first message"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = get_or_create_user_id(data)  # Get or create user ID
        chat_id = data.get('chat_id') or data.get('chatId')  # Support both formats
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get chat history from Firebase
        history = []
        is_new_chat = False
        
        if chat_id:
            # Existing chat - get history
            chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()
            if chat_doc.exists:
                messages = chat_doc.to_dict().get('messages', [])
                history = [f"{msg['sender']}: {msg['message']}" for msg in messages[-5:]]  # Last 5
            else:
                # Invalid chat_id, treat as new chat
                chat_id = None
        
        if not chat_id:
            # New chat - create it only now when user sends first message
            chat_id = str(uuid.uuid4())
            is_new_chat = True
        
        # Create prompt with Firebase history
        prompt = f"""
        You are a friendly agricultural medical assistant. Answer health questions naturally.

        Previous conversation: {history}

        User: {message}
        
        Respond helpfully but always remind users to consult doctors for serious concerns.
        """
        
        # Generate response
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        bot_response = response.text
        
        # Save to Firebase
        if is_new_chat:
            # Create new chat with first message exchange
            db.collection("users").document(user_id).collection("chats").document(chat_id).set({
                "createdAt": datetime.now(),
                "lastMessage": bot_response,
                "updatedAt": datetime.now(),
                "messages": [
                    {"sender": "user", "message": message, "timestamp": datetime.now()},
                    {"sender": "bot", "message": bot_response, "timestamp": datetime.now()}
                ]
            })
        else:
            # Update existing chat
            db.collection("users").document(user_id).collection("chats").document(chat_id).update({
                "lastMessage": bot_response,
                "updatedAt": datetime.now(),
                "messages": firestore.ArrayUnion([
                    {"sender": "user", "message": message, "timestamp": datetime.now()},
                    {"sender": "bot", "message": bot_response, "timestamp": datetime.now()}
                ])
            })
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'chat_id': chat_id,
            'user_id': user_id,  # Always return user_id
            'is_new_chat': is_new_chat
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Analyze image endpoint - handles both new and existing chats
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Image analysis with automatic chat management - creates chat only when user uploads first image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Get user data - support both form data and JSON
        form_data = dict(request.form)
        user_id = get_or_create_user_id(form_data)  # Get or create user ID
        chat_id = form_data.get('chat_id') or form_data.get('chatId')

        image_path = 'temp_image.jpg'
        image_file.save(image_path)

        # Predict using your model
        input_img = preprocess_image(image_path)
        prediction = disease_model.predict(input_img)

        CLASS_NAMES = list(FRIENDLY_LABELS.keys())
        predicted_index = np.argmax(prediction)
        predicted_key = CLASS_NAMES[predicted_index]
        predicted_label = FRIENDLY_LABELS[predicted_key]

        prompt = f"""
        A plant has been detected with the disease: {predicted_label}.
        Please explain what this disease is, how it affects the plant, and how a farmer can treat or prevent it.
        Keep it short and clear.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        # Handle Firebase chat history
        is_new_chat = False
        if chat_id:
            # Check if chat exists
            chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()
            if not chat_doc.exists:
                # Invalid chat_id, treat as new chat
                chat_id = None
                
        if not chat_id:
            # New chat - create it only now when user uploads first image
            chat_id = str(uuid.uuid4())
            is_new_chat = True

        # Prepare messages
        user_message = f"[Image Analysis] Uploaded plant image"
        bot_message = f"Disease detected: {predicted_label}\n\n{response.text}"
        
        # Save to Firebase
        if is_new_chat:
            # Create new chat with first image analysis
            db.collection("users").document(user_id).collection("chats").document(chat_id).set({
                "createdAt": datetime.now(),
                "lastMessage": bot_message,
                "updatedAt": datetime.now(),
                "messages": [
                    {"sender": "user", "message": user_message, "timestamp": datetime.now(), "type": "image"},
                    {"sender": "bot", "message": bot_message, "timestamp": datetime.now(), "type": "analysis"}
                ]
            })
        else:
            # Update existing chat
            db.collection("users").document(user_id).collection("chats").document(chat_id).update({
                "lastMessage": bot_message,
                "updatedAt": datetime.now(),
                "messages": firestore.ArrayUnion([
                    {"sender": "user", "message": user_message, "timestamp": datetime.now(), "type": "image"},
                    {"sender": "bot", "message": bot_message, "timestamp": datetime.now(), "type": "analysis"}
                ])
            })

        # Clean up temp file
        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'gemini_explanation': response.text,
            'chat_id': chat_id,
            'user_id': user_id,  # Always return user_id
            'is_new_chat': is_new_chat
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# History endpoint
@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Get user's consultation history"""
    try:
        # Validate user_id
        if not user_id or user_id in ['0', '', 'null', 'undefined']:
            return jsonify({'error': 'Invalid user ID provided'}), 400
            
        consultations = user_consultations.get(user_id, [])
        chats = chat_history.get(user_id, [])
        
        return jsonify({
            'success': True,
            'consultations': consultations[-5:],  # Last 5 consultations
            'recent_chats': chats[-10:]  # Last 10 chat messages
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Home endpoint
@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        'message': 'Agricultural Medical Chat API',
        'endpoints': {
            'POST /createUser': 'Create or get user ID',
            'GET /getUser/<user_id>': 'Get user information',
            'POST /chat': 'Chat with agricultural assistant (creates/continues chat)',
            'POST /analyze_image': 'Analyze plant disease images (creates/continues chat)',
            'GET /history/<user_id>': 'Get user history',
            'GET /health': 'Health check',
            'POST /addCrop': 'Add crop data to Firebase',
            'PUT /updateCrop': 'Update crop data in Firebase',
            'DELETE /deleteCrop': 'Delete crop data from Firebase',
            'GET /getCrops': 'Get crops from Firebase',
            'GET /getSuggestions': 'Get farming suggestions based on crops',
            'GET /getChats': 'Get all user chat history',
            'GET /getChat': 'Get specific chat conversation'
        }
    })

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'users': len(chat_history),
        'consultations': sum(len(v) for v in user_consultations.values())
    })

# Crop management endpoints
@app.route('/addCrop', methods=['POST'])
def add_crop():
    try:
        data = request.get_json()
        user_id = get_or_create_user_id(data)  # Get or create user ID
        crop_data = data.get('cropData')
        
        if not crop_data:
            return jsonify({"error": "Missing cropData"}), 400

        crop_id = str(uuid.uuid4())
        crop_data["timestamp"] = datetime.now()

        db.collection("users").document(user_id).collection("crops").document(crop_id).set(crop_data)

        return jsonify({
            "message": "Crop added successfully", 
            "cropId": crop_id,
            "userId": user_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/updateCrop', methods=['PUT'])
def update_crop():
    try:
        data = request.get_json()
        user_id = get_or_create_user_id(data)  # Get or create user ID
        crop_id = data.get('cropId')
        crop_data = data.get('cropData')
        
        if not crop_id or not crop_data:
            return jsonify({"error": "Missing cropId or cropData"}), 400

        crop_data["updatedAt"] = datetime.now()

        db.collection("users").document(user_id).collection("crops").document(crop_id).update(crop_data)

        return jsonify({
            "message": "Crop updated successfully",
            "userId": user_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/deleteCrop', methods=['DELETE'])
def delete_crop():
    try:
        # Support both query params and JSON body
        if request.is_json:
            data = request.get_json()
            user_id = get_or_create_user_id(data)
            crop_id = data.get("cropId")
        else:
            user_id = request.args.get("userId")
            crop_id = request.args.get("cropId")
            
            # Validate user_id if from query params
            if not user_id or user_id in ['0', '', 'null', 'undefined']:
                return jsonify({"error": "Invalid user ID provided"}), 400

        if not crop_id:
            return jsonify({"error": "Missing cropId"}), 400

        db.collection("users").document(user_id).collection("crops").document(crop_id).delete()

        return jsonify({
            "message": "Crop deleted successfully",
            "userId": user_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getCrops', methods=['GET'])
def get_crops():
    try:
        user_id = request.args.get('userId')
        
        # Validate user_id
        if not user_id or user_id in ['0', '', 'null', 'undefined']:
            return jsonify({"error": "Invalid user ID provided"}), 400

        crops_ref = db.collection("users").document(user_id).collection("crops")
        crops = crops_ref.stream()

        crop_list = []
        for crop in crops:
            crop_data = crop.to_dict()
            crop_data["id"] = crop.id
            crop_list.append(crop_data)

        return jsonify({
            "crops": crop_list,
            "userId": user_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home Screen suggestion endpoint
@app.route('/getSuggestions', methods=['GET'])
def get_suggestions():
    try:
        user_id = request.args.get("userId")
        
        # Validate user_id
        if not user_id or user_id in ['0', '', 'null', 'undefined']:
            return jsonify({"error": "Invalid user ID provided"}), 400

        # Fetch crops
        crops_ref = db.collection("users").document(user_id).collection("crops").stream()
        crops = [doc.to_dict() for doc in crops_ref]

        if not crops:
            return jsonify({"error": "No crops found for this user"}), 404

        # Calculate days since sowing for each crop
        from datetime import datetime, timedelta
        current_date = datetime.now()
        
        crop_details = []
        for crop in crops:
            try:
                sowed_date = datetime.strptime(crop['sowedDate'], '%Y-%m-%d')
                days_old = (current_date - sowed_date).days
                crop_details.append({
                    'name': crop['name'],
                    'area': crop['area'],
                    'days_old': days_old,
                    'sowed_date': crop['sowedDate']
                })
            except (ValueError, KeyError):
                # Skip crops with invalid dates
                continue

        if not crop_details:
            return jsonify({"error": "No valid crops with proper dates found"}), 404

        # Enhanced prompt for specific, actionable recommendations
        crop_lines = [
            f"- {crop['name']}: {crop['area']} acres, {crop['days_old']} days old (sown {crop['sowed_date']})"
            for crop in crop_details
        ]
        
        prompt = f"""Based on these crops and their growth stages:
{chr(10).join(crop_lines)}

Current date: {current_date.strftime('%Y-%m-%d')}

Provide exactly 4 specific farming recommendations in this priority order:
1. URGENT WATERING: Which crop needs water most urgently and when?
2. CRITICAL CARE: Which crop needs immediate attention (pest/disease/nutrition)?
3. UPCOMING TASK: What's the next important farming activity needed?
4. WEATHER ACTION: What to do based on current season/weather?

Format each as a short, actionable card (max 25 words). Start each with an action verb. Include specific crop names when relevant.

Examples:
- "Water tomatoes immediately - 28 days old, critical growth stage"
- "Check wheat for aphids - inspect leaves daily this week"
- "Apply nitrogen fertilizer to corn in 3 days"
- "Prepare rice fields for monsoon - clean drainage channels"
"""

        # Call Gemini
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        raw_output = response.text.strip()

        # Parse suggestions more intelligently
        suggestions = []
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        
        for line in lines:
            # Remove numbering, bullets, and common prefixes
            cleaned = line.strip('1234567890.-•* ').strip()
            if cleaned and len(cleaned) > 10:  # Filter out very short lines
                suggestions.append(cleaned)

        # Ensure we have exactly 4 suggestions
        while len(suggestions) < 4:
            suggestions.append("Monitor crop health daily")
        
        suggestions = suggestions[:4]

        # Structure response with priority categories
        return jsonify({
            "suggestions": {
                "urgent_watering": {
                    "text": suggestions[0],
                    "priority": "high",
                    "category": "watering"
                },
                "critical_care": {
                    "text": suggestions[1],
                    "priority": "high", 
                    "category": "care"
                },
                "upcoming_task": {
                    "text": suggestions[2],
                    "priority": "medium",
                    "category": "planning"
                },
                "weather_action": {
                    "text": suggestions[3],
                    "priority": "medium",
                    "category": "weather"
                }
            },
            "generated_at": current_date.isoformat(),
            "total_crops": len(crop_details),
            "userId": user_id
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to generate suggestions: {str(e)}"}), 500

# Chat history management endpoints
@app.route('/getChats', methods=['GET'])
def get_chats():
    """Get all chats for user"""
    try:
        user_id = request.args.get('userId')
        
        # Validate user_id
        if not user_id or user_id in ['0', '', 'null', 'undefined']:
            return jsonify({"error": "Invalid user ID provided"}), 400
            
        chats = db.collection("users").document(user_id).collection("chats")\
                .order_by("createdAt", direction=firestore.Query.DESCENDING).stream()
        
        chat_list = []
        for chat in chats:
            data = chat.to_dict()
            created_at = data.get("createdAt")
            
            # Convert Firestore timestamp to ISO format string if not None
            if created_at:
                created_at_str = created_at.isoformat()
            else:
                created_at_str = None
                
            chat_list.append({
                "chatId": chat.id,
                "lastMessage": data.get("lastMessage", ""),
                "createdAt": created_at_str,
                "updatedAt": data.get("updatedAt", created_at_str)
            })
            
        return jsonify({
            "chats": chat_list,
            "userId": user_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getChat', methods=['GET'])
def get_chat():
    """Get specific chat conversation"""
    try:
        user_id = request.args.get('userId')
        chat_id = request.args.get('chatId')

        # Validate user_id
        if not user_id or user_id in ['0', '', 'null', 'undefined']:
            return jsonify({"error": "Invalid user ID provided"}), 400
            
        if not chat_id:
            return jsonify({"error": "Missing chatId"}), 400

        # Fetch the chat document
        chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)
        chat_doc = chat_ref.get()

        if not chat_doc.exists:
            return jsonify({"error": "Chat not found"}), 404

        chat_data = chat_doc.to_dict()
        messages = chat_data.get("messages", [])

        # Convert timestamps to ISO format for frontend
        for msg in messages:
            if "timestamp" in msg:
                msg["timestamp"] = msg["timestamp"].isoformat()

        return jsonify({
            "chatId": chat_id,
            "userId": user_id,
            "createdAt": chat_data.get("createdAt").isoformat() if chat_data.get("createdAt") else None,
            "updatedAt": chat_data.get("updatedAt").isoformat() if chat_data.get("updatedAt") else None,
            "messages": messages
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Delete all chats endpoint
@app.route('/deleteAllChats', methods=['DELETE'])
def delete_all_chats():
    """Delete all chats for a given user"""
    try:
        user_id = request.args.get('userId')

        # Validate userId
        if not user_id or user_id in ['0', '', 'null', 'undefined']:
            return jsonify({"error": "Invalid user ID provided"}), 400

        # Reference to the user's chat collection
        chat_collection_ref = db.collection("users").document(user_id).collection("chats")

        # Get all chat documents
        chat_docs = chat_collection_ref.stream()

        deleted_count = 0
        for doc in chat_docs:
            doc.reference.delete()
            deleted_count += 1

        return jsonify({
            "message": f"Deleted {deleted_count} chat(s) for user {user_id}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    

if __name__ == '__main__':
    print("🌾 Agricultural Medical Chat API")
    print("=" * 35)
    print("📋 Main Endpoints:")
    print("  POST /createUser - Create or get user ID")
    print("  GET /getUser/<user_id> - Get user information")
    print("  POST /chat - Chat with assistant (auto-manages chats)")
    print("  POST /analyze_image - Analyze plant images (auto-manages chats)")
    print("  GET /getChats - Get all user's chat history")
    print("  GET /getChat - Get specific chat conversation")
    print("  GET /health - Health check")
    print("\n🌱 Crop Management:")
    print("  POST /addCrop - Add crop data")
    print("  PUT /updateCrop - Update crop data")
    print("  DELETE /deleteCrop - Delete crop data")
    print("  GET /getCrops - Get user's crops")
    print("  GET /getSuggestions - Get farming suggestions")
    print("\n🚀 Starting server...")
    
    app.run(host='0.0.0.0', port=5000, debug=True)