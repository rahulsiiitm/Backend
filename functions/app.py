import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import random
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
import json

# Load environment variables from .env file
load_dotenv()  # Add this line

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firebase
firebase_key = os.getenv("FIREBASE_KEY")
cred = credentials.Certificate(json.loads(firebase_key))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model
disease_model = tf.keras.models.load_model("functions/models/plant_disease_model.keras")

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




def generate_farming_suggestions_with_gemini(crops, weather_data):
    """Generate farming suggestions using Gemini AI based on user's actual crops and weather"""
    try:
        # Prepare crop information
        crop_info = []
        for crop in crops:
            crop_info.append(f"- {crop['name']} ({crop.get('type', 'unknown type')}, planted {crop['days_old']} days ago)")
        
        crops_text = "\n".join(crop_info)
        
        # Prepare weather information
        if weather_data:
            current_weather = weather_data['current']
            weather_text = f"""
Current Weather:
- Temperature: {current_weather['temperature']}°C (feels like {current_weather['feels_like']}°C)
- Humidity: {current_weather['humidity']}%
- Weather: {current_weather['description']}
- Wind Speed: {current_weather['wind_speed']} m/s
- Pressure: {current_weather['pressure']} hPa

Forecast (next 24 hours):
"""
            for i, forecast in enumerate(weather_data['forecast'][:4]):
                weather_text += f"- {forecast['date']}: {forecast['temp']}°C, {forecast['description']}, Rain: {forecast['rain']}mm\n"
        else:
            weather_text = "Weather data not available"

        prompt = f"""
You are an expert agricultural advisor. Based on the farmer's crops and current weather conditions, provide 4 practical farming suggestions.

Farmer's Crops:
{crops_text}

{weather_text}

Please provide exactly 4 specific, actionable farming suggestions. Each suggestion should:
1. Be practical and immediately actionable
2. Consider the current weather conditions
3. Be specific to the crops the farmer is growing
4. Include the crop name in the suggestion
5. Be concise (1-2 sentences each)

Format your response as a JSON array with 4 objects, each having:
- "text": the suggestion text
- "category": one of ["irrigation", "fertilizer", "protection", "care", "harvesting", "pest_control"]
- "crop": the specific crop name mentioned
- "priority": one of ["high", "medium", "low"]

Example format:
[
  {{
    "text": "Water your wheat early morning - temperature reaching 35°C today will stress the plants",
    "category": "irrigation", 
    "crop": "wheat",
    "priority": "high"
  }}
]
"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Try to parse JSON response
        import json
        try:
            # Clean the response text to extract JSON
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            suggestions = json.loads(response_text)
            
            # Validate the response format
            if isinstance(suggestions, list) and len(suggestions) >= 4:
                return suggestions[:4]  # Return only first 4
            else:
                raise ValueError("Invalid suggestion format")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse Gemini JSON response: {e}")
            # Fallback to text parsing
            return parse_text_suggestions(response.text, crops)
            
    except Exception as e:
        print(f"Error generating suggestions with Gemini: {e}")
        return generate_fallback_suggestions(crops, weather_data)

def parse_text_suggestions(text, crops):
    """Parse text suggestions if JSON parsing fails"""
    lines = text.split('\n')
    suggestions = []
    current_suggestion = ""
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('*'):
            if any(crop['name'].lower() in line.lower() for crop in crops):
                if current_suggestion:
                    suggestions.append({
                        "text": current_suggestion,
                        "category": "care",
                        "crop": "general",
                        "priority": "medium"
                    })
                current_suggestion = line
            elif current_suggestion:
                current_suggestion += " " + line
    
    if current_suggestion:
        suggestions.append({
            "text": current_suggestion,
            "category": "care", 
            "crop": "general",
            "priority": "medium"
        })
    
    return suggestions[:4]

def generate_fallback_suggestions(crops, weather_data):
    """Generate basic fallback suggestions if Gemini fails"""
    suggestions = []
    temp = weather_data['current']['temperature'] if weather_data else 25
    humidity = weather_data['current']['humidity'] if weather_data else 65
    
    for i, crop in enumerate(crops[:4]):
        crop_name = crop['name']
        days_old = crop['days_old']
        
        if temp > 30:
            suggestion = f"Provide shade or extra water to your {crop_name} - high temperature ({temp}°C) can stress the plants"
            category = "protection"
            priority = "high"
        elif humidity > 80:
            suggestion = f"Check your {crop_name} for fungal diseases - high humidity ({humidity}%) increases disease risk"
            category = "protection"
            priority = "medium"
        elif days_old > 60:
            suggestion = f"Consider harvesting your {crop_name} soon - it's been {days_old} days since planting"
            category = "harvesting"
            priority = "medium"
        else:
            suggestion = f"Monitor your {crop_name} growth - apply balanced fertilizer if needed after {days_old} days"
            category = "fertilizer"
            priority = "medium"
        
        suggestions.append({
            "text": suggestion,
            "category": category,
            "crop": crop_name,
            "priority": priority
        })
    
    return suggestions

def generate_daily_suggestion_with_gemini(crops, weather_data):
    """Generate a single daily suggestion using Gemini"""
    try:
        # Prepare crop information
        crop_names = [crop['name'] for crop in crops]
        crops_text = ", ".join(crop_names)
        
        # Prepare weather information
        if weather_data:
            current_weather = weather_data['current']
            weather_text = f"Temperature: {current_weather['temperature']}°C, Humidity: {current_weather['humidity']}%, Weather: {current_weather['description']}"
        else:
            weather_text = "Weather data not available"

        prompt = f"""
You are a friendly agricultural advisor. Create ONE motivational daily suggestion for a farmer.

Farmer's crops: {crops_text}
Today's weather: {weather_text}

Create a short, friendly daily suggestion with:
1. A catchy heading with appropriate emoji
2. A brief, actionable message (1-2 sentences)
3. Mention one of their specific crops
4. Consider today's weather conditions

Format as JSON:
{{
  "heading": "Good morning message with emoji",
  "body": "Specific actionable advice mentioning their crop"
}}

Keep it encouraging and practical!
"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Try to parse JSON response
        import json
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            suggestion = json.loads(response_text)
            
            if 'heading' in suggestion and 'body' in suggestion:
                return suggestion
            else:
                raise ValueError("Invalid suggestion format")
                
        except (json.JSONDecodeError, ValueError):
            # Fallback to simple suggestion
            selected_crop = random.choice(crops)['name']
            temp = weather_data['current']['temperature'] if weather_data else 25
            
            return {
                "heading": "Good morning farmer! 🌱",
                "body": f"Check on your {selected_crop} today - with {temp}°C weather, it's a great day for farming!"
            }
            
    except Exception as e:
        print(f"Error generating daily suggestion: {e}")
        # Simple fallback
        selected_crop = random.choice(crops)['name']
        return {
            "heading": "Farm check time! 🚜",
            "body": f"How's your {selected_crop} doing today? A quick inspection never hurts!"
        }









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
        You are a friendly agricultural medical assistant. Answer health questions naturally, engage with the user but keep the text short and clear.

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










@app.route('/getDailySuggestion', methods=['GET'])
def get_daily_suggestion():
    try:
        user_id = request.args.get("userId")
        lat = request.args.get('lat', 27.1767, type=float)
        lon = request.args.get('lon', 78.0081, type=float)
        
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({"error": "Valid userId is required"}), 400

        # Update user activity
        update_user_activity(user_id)

        # Get crops from Firebase
        crops_ref = db.collection("users").document(user_id).collection("crops").stream()
        crops = []
        current_date = datetime.now()
        
        for doc in crops_ref:
            crop_data = doc.to_dict()
            try:
                sowed_date = datetime.strptime(crop_data.get('sowedDate', ''), '%Y-%m-%d')
                days_old = (current_date - sowed_date).days
                crops.append({
                    'name': crop_data.get('name', 'Unknown Crop'),
                    'type': crop_data.get('type', ''),
                    'days_old': days_old
                })
            except (ValueError, KeyError):
                crops.append({
                    'name': crop_data.get('name', 'Unknown Crop'),
                    'type': crop_data.get('type', ''),
                    'days_old': 30
                })

        if not crops:
            return jsonify({"error": "No crops found for this user. Please add crops first."}), 404

        # Get weather data
        weather_data = get_weather_data(lat, lon)
        
        # Generate daily suggestion using Gemini
        suggestion = generate_daily_suggestion_with_gemini(crops, weather_data)

        return jsonify({
            "success": True,
            "suggestion": suggestion
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to generate daily suggestion: {str(e)}"
        }), 500

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

        # Get crops from Firebase
        crops_ref = db.collection("users").document(user_id).collection("crops").stream()
        crops = []
        current_date = datetime.now()
        
        for doc in crops_ref:
            crop_data = doc.to_dict()
            try:
                # Calculate days old
                sowed_date = datetime.strptime(crop_data.get('sowedDate', ''), '%Y-%m-%d')
                days_old = (current_date - sowed_date).days
                
                crops.append({
                    'id': doc.id,
                    'name': crop_data.get('name', 'Unknown Crop'),
                    'type': crop_data.get('type', ''),
                    'area': crop_data.get('area', ''),
                    'days_old': days_old,
                    'sowed_date': crop_data.get('sowedDate', '')
                })
            except (ValueError, KeyError):
                # If date parsing fails, add with default days_old
                crops.append({
                    'id': doc.id,
                    'name': crop_data.get('name', 'Unknown Crop'),
                    'type': crop_data.get('type', ''),
                    'area': crop_data.get('area', ''),
                    'days_old': 30,  # Default
                    'sowed_date': crop_data.get('sowedDate', '')
                })

        if not crops:
            return jsonify({"error": "No crops found for this user. Please add crops first."}), 404

        # Get weather data
        weather_data = get_weather_data(lat, lon)
        
        # Generate suggestions using Gemini
        suggestions = generate_farming_suggestions_with_gemini(crops, weather_data)
        
        # Format suggestions for response
        formatted_suggestions = {}
        suggestion_keys = ['first', 'second', 'third', 'fourth']
        
        for i, suggestion in enumerate(suggestions):
            key = suggestion_keys[i] if i < len(suggestion_keys) else f'suggestion_{i+1}'
            formatted_suggestions[key] = suggestion

        return jsonify({
            "success": True,
            "suggestions": formatted_suggestions
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to generate suggestions: {str(e)}"
        }), 500
    












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
