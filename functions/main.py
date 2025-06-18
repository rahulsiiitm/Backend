from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import requests
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()  # Add this line

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model
disease_model = load_model("models/plant_disease_model.keras")

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

# Configure APIs
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')

if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found")
    exit(1)
if not OPENWEATHER_API_KEY:
    print("❌ Error: OPENWEATHER_API_KEY not found")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
print("✅ APIs configured successfully")

# Utility functions
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def validate_user_id(user_id):
    """Validate that user_id is provided and not empty"""
    if not user_id or user_id.strip() == '' or user_id in ['null', 'undefined']:
        return False
    return True

def update_user_activity(user_id):
    try:
        user_ref = db.collection("users").document(user_id)
        user_ref.set({"lastActive": datetime.now()}, merge=True)
    except Exception as e:
        app.logger.warning(f"Could not update user activity for {user_id}: {e}")

def get_weather_data(lat, lon):
    """Fetch current weather and 5-day forecast"""
    try:
        # Current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        current_response.raise_for_status()
        current_data = current_response.json()
        
        # Forecast (next 5 days in 3-hour intervals)
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        return {
            'current': {
                'temperature': current_data['main']['temp'],
                'humidity': current_data['main']['humidity'],
                'description': current_data['weather'][0]['description'],
                'wind_speed': current_data['wind']['speed'],
                'pressure': current_data['main']['pressure'],
                'feels_like': current_data['main']['feels_like']
            },
            'forecast': [
                {
                    'date': item['dt_txt'],
                    'temp': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'description': item['weather'][0]['description'],
                    'rain': item.get('rain', {}).get('3h', 0)
                }
                for item in forecast_data['list'][:8]  # Next 24 hours
            ]
        }

    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return None

@app.route('/weather', methods=['GET'])
def get_weather():
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)

        if lat is None or lon is None:
            return jsonify({
                'success': False,
                'error': 'Latitude and longitude are required.'
            }), 400
        
        weather_data = get_weather_data(lat, lon)
        if not weather_data:
            return jsonify({'success': False, 'error': 'Failed to fetch weather data'}), 500
            
        return jsonify({
            'success': True,
            'weather': weather_data,
            'location': {'lat': lat, 'lon': lon}
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Chat endpoint
@app.route('/chat', methods=['POST'])
def medical_chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id') or data.get('userId')
        chat_id = data.get('chat_id') or data.get('chatId')
        
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({'error': 'Valid user_id is required'}), 400
            
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Update user activity
        update_user_activity(user_id)
        
        # Get chat history
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
        
        # Create prompt
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
            'chat_id': chat_id,
            'user_id': user_id,
            'is_new_chat': is_new_chat
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Image analysis endpoint
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        form_data = dict(request.form)
        user_id = form_data.get('user_id') or form_data.get('userId')
        chat_id = form_data.get('chat_id') or form_data.get('chatId')

        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({'error': 'Valid user_id is required'}), 400

        # Update user activity
        update_user_activity(user_id)

        image_path = 'temp_image.jpg'
        image_file.save(image_path)

        # Predict disease
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

        # Handle chat creation/update
        is_new_chat = False
        if chat_id:
            chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()
            if not chat_doc.exists:
                chat_id = None
                
        if not chat_id:
            chat_id = str(uuid.uuid4())
            is_new_chat = True

        user_message = f"[Image Analysis] Uploaded plant image"
        bot_message = f"Disease detected: {predicted_label}\n\n{response.text}"
        
        message_data = [
            {"sender": "user", "message": user_message, "timestamp": datetime.now(), "type": "image"},
            {"sender": "bot", "message": bot_message, "timestamp": datetime.now(), "type": "analysis"}
        ]
        
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

        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'gemini_explanation': response.text,
            'chat_id': chat_id,
            'user_id': user_id,
            'is_new_chat': is_new_chat
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Crop management endpoints
@app.route('/addCrop', methods=['POST'])
def add_crop():
    try:
        data = request.get_json()
        user_id = data.get('user_id') or data.get('userId')
        crop_data_list = data.get('cropData')

        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid user_id is required"}), 400

        if not crop_data_list or not isinstance(crop_data_list, list):
            return jsonify({"error": "cropData must be a non-empty list"}), 400

        # Update user activity
        update_user_activity(user_id)

        added_crops = []
        for crop_data in crop_data_list:
            crop_id = str(uuid.uuid4())
            crop_data["timestamp"] = datetime.now().isoformat()
            db.collection("users").document(user_id).collection("crops").document(crop_id).set(crop_data)
            added_crops.append({"cropId": crop_id, "data": crop_data})

        return jsonify({
            "message": "Crop(s) added successfully",
            "userId": user_id,
            "cropsAdded": added_crops
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getCrops', methods=['GET'])
def get_crops():
    try:
        user_id = request.args.get('userId')
        
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400

        # Update user activity
        update_user_activity(user_id)

        crops_ref = db.collection("users").document(user_id).collection("crops")
        crops = crops_ref.stream()

        crop_list = []
        for crop in crops:
            crop_data = crop.to_dict()
            crop_list.append({
                "id": crop.id,
                "name": str(crop_data.get('name', '')),
                "type": str(crop_data.get('type', '')),
                "plantedDate": str(crop_data.get('sowedDate') or crop_data.get('plantedDate', '')),
                "area": str(crop_data.get('area', '')),
            })

        return jsonify({
            "crops": crop_list,
            "userId": user_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Enhanced suggestions with weather integration
@app.route('/getSuggestions', methods=['GET'])
def get_suggestions():
    try:
        user_id = request.args.get("userId")
        lat = request.args.get('lat', 27.1767, type=float)
        lon = request.args.get('lon', 78.0081, type=float)
        
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400

        # Update user activity
        update_user_activity(user_id)

        # Get crops
        crops_ref = db.collection("users").document(user_id).collection("crops").stream()
        crops = [doc.to_dict() for doc in crops_ref]

        if not crops:
            return jsonify({"error": "No crops found for this user"}), 404

        # Get weather data
        weather_data = get_weather_data(lat, lon)
        
        # Process crops
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
                continue

        if not crop_details:
            return jsonify({"error": "No valid crops found"}), 404

        # Create enhanced prompt with weather data
        crop_lines = [f"- {crop['name']}: {crop['area']} acres, {crop['days_old']} days old" for crop in crop_details]
        
        weather_info = ""
        if weather_data:
            weather_info = f"""
Current Weather:
- Temperature: {weather_data['current']['temperature']}°C (feels like {weather_data['current']['feels_like']}°C)
- Humidity: {weather_data['current']['humidity']}%
- Condition: {weather_data['current']['description']}
- Wind: {weather_data['current']['wind_speed']} m/s

24-hour Forecast: {weather_data['forecast'][0]['description']}, {weather_data['forecast'][0]['temp']}°C
"""

        prompt = f"""Based on these crops and current weather conditions:

CROPS:
{chr(10).join(crop_lines)}

{weather_info}

Current date: {current_date.strftime('%Y-%m-%d')}

Provide exactly 4 specific farming recommendations considering BOTH crop stages AND current weather:
1. WEATHER-BASED URGENT ACTION: Most critical action needed based on current weather
2. CROP CARE: Which crop needs immediate attention based on growth stage
3. IRRIGATION: Watering advice based on weather conditions and crop needs
4. PROTECTION: Weather protection or upcoming weather preparation

Format each as a short, actionable card (max 25 words). Start with action verbs. Include specific crop names.

Examples:
- "Cover tomatoes tonight - temperature dropping to 15°C, protect young plants"
- "Increase wheat watering - low humidity (45%) and high temperature stress"
- "Apply fungicide to rice - high humidity (80%) increases disease risk"
- "Prepare drainage for corn - heavy rain forecast in 24 hours"
"""

        # Generate suggestions
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        raw_output = response.text.strip()

        # Parse suggestions
        suggestions = []
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        
        for line in lines:
            cleaned = line.strip('1234567890.-•* ').strip()
            if cleaned and len(cleaned) > 10:
                suggestions.append(cleaned)

        while len(suggestions) < 4:
            suggestions.append("Monitor crop health daily")
        
        suggestions = suggestions[:4]

        return jsonify({
            "suggestions": {
                "weather_urgent": {
                    "text": suggestions[0],
                    "priority": "high",
                    "category": "weather"
                },
                "crop_care": {
                    "text": suggestions[1],
                    "priority": "high", 
                    "category": "care"
                },
                "irrigation": {
                    "text": suggestions[2],
                    "priority": "medium",
                    "category": "watering"
                },
                "protection": {
                    "text": suggestions[3],
                    "priority": "medium",
                    "category": "protection"
                }
            },
            "weather": weather_data,
            "generated_at": current_date.isoformat(),
            "total_crops": len(crop_details),
            "userId": user_id
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate suggestions: {str(e)}"}), 500

# Chat history endpoints
@app.route('/getChats', methods=['GET'])
def get_chats():
    try:
        user_id = request.args.get('userId')
        
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400

        # Update user activity
        update_user_activity(user_id)
            
        chats = db.collection("users").document(user_id).collection("chats")\
                .order_by("createdAt", direction=firestore.Query.DESCENDING).stream()
        
        chat_list = []
        for chat in chats:
            data = chat.to_dict()
            created_at = data.get("createdAt")
            chat_list.append({
                "chatId": chat.id,
                "lastMessage": data.get("lastMessage", ""),
                "createdAt": created_at.isoformat() if created_at else None,
                "updatedAt": data.get("updatedAt", created_at).isoformat() if data.get("updatedAt") else None
            })
            
        return jsonify({"chats": chat_list, "userId": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getChat', methods=['GET'])
def get_chat():
    try:
        user_id = request.args.get('userId')
        chat_id = request.args.get('chatId')

        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400
            
        if not chat_id:
            return jsonify({"error": "Missing chatId"}), 400

        # Update user activity
        update_user_activity(user_id)

        chat_doc = db.collection("users").document(user_id).collection("chats").document(chat_id).get()

        if not chat_doc.exists:
            return jsonify({"error": "Chat not found"}), 404

        chat_data = chat_doc.to_dict()
        messages = chat_data.get("messages", [])

        for msg in messages:
            if "timestamp" in msg:
                msg["timestamp"] = msg["timestamp"].isoformat()

        return jsonify({
            "chatId": chat_id,
            "userId": user_id,
            "createdAt": chat_data.get("createdAt").isoformat() if chat_data.get("createdAt") else None,
            "updatedAt": chat_data.get("updatedAt").isoformat() if chat_data.get("updatedAt") else None,
            "messages": messages
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Additional endpoints
@app.route('/updateCrop', methods=['PUT'])
def update_crop():
    try:
        data = request.get_json()
        user_id = data.get('user_id') or data.get('userId')
        crop_id = data.get('cropId')
        crop_data = data.get('cropData')
        
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid user_id is required"}), 400
        
        if not crop_id or not crop_data:
            return jsonify({"error": "Missing cropId or cropData"}), 400

        # Update user activity
        update_user_activity(user_id)

        crop_data["updatedAt"] = datetime.now()
        db.collection("users").document(user_id).collection("crops").document(crop_id).update(crop_data)

        return jsonify({"message": "Crop updated successfully", "userId": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/deleteCrop', methods=['DELETE'])
def delete_crop():
    try:
        if request.is_json:
            data = request.get_json()
            user_id = data.get('user_id') or data.get('userId')
            crop_id = data.get("cropId")
        else:
            user_id = request.args.get("userId")
            crop_id = request.args.get("cropId")
            
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400

        if not crop_id:
            return jsonify({"error": "Missing cropId"}), 400

        # Update user activity
        update_user_activity(user_id)

        db.collection("users").document(user_id).collection("crops").document(crop_id).delete()
        return jsonify({"message": "Crop deleted successfully", "userId": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/deleteAllChats', methods=['DELETE'])
def delete_all_chats():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_id = data.get('userId')
        
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400

        # Update user activity
        update_user_activity(user_id)

        chat_docs = db.collection("users").document(user_id).collection("chats").stream()
        deleted_count = 0
        for doc in chat_docs:
            doc.reference.delete()
            deleted_count += 1

        return jsonify({
            "success": True,
            "message": f"Deleted {deleted_count} chat(s)",
            "deletedCount": deleted_count
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Health and info endpoints
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Agricultural Advisory System',
        'version': '2.1',
        'note': 'This API expects user_id from existing login system',
        'endpoints': {
            'POST /chat': 'Chat with assistant (requires user_id)',
            'POST /analyze_image': 'Analyze plant images (requires user_id)',
            'GET /weather': 'Get weather data',
            'GET /getSuggestions': 'Get weather-based farming suggestions (requires userId)',
            'POST /addCrop': 'Add crops (requires user_id)',
            'GET /getCrops': 'Get crops (requires userId)',
            'GET /getChats': 'Get chat history (requires userId)',
            'DELETE /deleteAllChats': 'Delete all chats (requires userId)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'weather_api': 'connected' if OPENWEATHER_API_KEY else 'missing'
    })

if __name__ == '__main__':
    print("🌾 Agricultural Advisory System (Login System Integration)")
    print("=" * 60)
    print("📋 Key Features:")
    print("  • Plant disease detection")
    print("  • Weather-based farming suggestions")
    print("  • Agricultural chatbot")
    print("  • Crop management")
    print("  • Real-time weather data")
    print("\n🔑 Authentication:")
    print("  • Expects user_id from existing login system")
    print("  • No user creation - uses Firebase Auth user IDs")
    print("\n🚀 Starting server...")
    
    app.run(host='0.0.0.0', port=5000, debug=True)