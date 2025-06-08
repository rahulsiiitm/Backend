from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import os
from datetime import datetime
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firebase App
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)

# Firestore DB client
db = firestore.client()


#testmessage

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

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # match training size
    img_array = image.img_to_array(img)  # shape: (128, 128, 3)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)
    return img_array

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

@app.route('/chat', methods=['POST'])
def medical_chat():
    """Simple medical chat"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'user1')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get chat history for this user
        if user_id not in chat_history:
            chat_history[user_id] = []
        
        # Create simple prompt
        prompt = f"""
        You are a friendly medical assistant. Answer health questions naturally.
        
        Previous conversation: {chat_history[user_id][-4:]}  # Last 4 messages
        
        User: {message}
        
        Respond helpfully but always remind users to consult doctors for serious concerns.
        """
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        bot_response = response.text
        
        # Save to memory
        chat_history[user_id].append(f"User: {message}")
        chat_history[user_id].append(f"Bot: {bot_response}")
        
        return jsonify({
            'success': True,
            'response': bot_response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_symptoms():
    """Analyze symptoms with structured response"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        user_id = data.get('user_id', 'user1')
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Simple analysis prompt
        prompt = f"""
        Analyze these symptoms and respond in JSON format:
        Symptoms: {symptoms}
        
        {{
          "conditions": ["condition1", "condition2"],
          "severity": "Low/Medium/High",
          "advice": "what to do next",
          "disclaimer": "Always consult a doctor for proper diagnosis"
        }}
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Try to parse JSON
        try:
            result = json.loads(response.text.strip('```json').strip('```'))
        except:
            result = {
                "conditions": ["Please consult a doctor"],
                "severity": "Unknown",
                "advice": "Get medical attention",
                "disclaimer": "Always consult a doctor for proper diagnosis"
            }
        
        # Save consultation
        if user_id not in user_consultations:
            user_consultations[user_id] = []
        
        consultation = {
            'symptoms': symptoms,
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        }
        user_consultations[user_id].append(consultation)
        
        return jsonify({
            'success': True,
            'analysis': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

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
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'gemini_explanation': response.text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Get user's consultation history"""
    try:
        consultations = user_consultations.get(user_id, [])
        chats = chat_history.get(user_id, [])
        
        return jsonify({
            'success': True,
            'consultations': consultations[-5:],  # Last 5 consultations
            'recent_chats': chats[-10:]  # Last 10 chat messages
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        'message': 'Simple Medical Chat API',
        'endpoints': {
            'POST /chat': 'Chat with medical assistant',
            'POST /analyze': 'Analyze symptoms',
            'POST /analyze_image': 'Analyze medical images',
            'GET /history/<user_id>': 'Get user history'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'users': len(chat_history),
        'consultations': sum(len(v) for v in user_consultations.values())
    })

@app.route('/addCrop', methods=['POST'])
def add_crop():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        crop_data = data.get('cropData')
        if not user_id or not crop_data:
            return jsonify({"error": "Missing userId or cropData"}), 400

        crop_id = str(uuid.uuid4())
        crop_data["timestamp"] = datetime.now()

        db.collection("users").document(user_id).collection("crops").document(crop_id).set(crop_data)

        return jsonify({"message": "Crop added successfully", "cropId": crop_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/deleteCrop', methods=['DELETE'])
def delete_crop():
    try:
        user_id = request.args.get("userId")
        crop_id = request.args.get("cropId")

        if not user_id or not crop_id:
            return jsonify({"error": "Missing userId or cropId"}), 400

        db.collection("plants").document(user_id).collection("crops").document(crop_id).delete()

        return jsonify({"message": "Crop deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/getCrops', methods=['GET'])
def get_crops():
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({"error": "Missing userId"}), 400

        crops_ref = db.collection("users").document(user_id).collection("crops")
        crops = crops_ref.stream()

        crop_list = []
        for crop in crops:
            crop_data = crop.to_dict()
            crop_data["id"] = crop.id
            crop_list.append(crop_data)

        return jsonify({"crops": crop_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return "Firebase is connected and the server is running!"


if __name__ == '__main__':
    print("🏥 Simple Medical Chat API")
    print("=" * 35)
    print("📋 Endpoints:")
    print("  POST /chat - Medical chat")
    print("  POST /analyze - Symptom analysis")
    print("  POST /analyze_image - Image analysis")
    print("  GET /history/<user_id> - User history")
    print("  GET /health - Health check")
    print("  POST /addCrop - Add crop data to Firebase")
    print("  DELETE /deleteCrop - Delete crop data from Firebase")
    print("  GET /getCrops - Get crops from Firebase")
    print("\n🚀 Starting server...")
    
    app.run(host='0.0.0.0', port=5000, debug=True)