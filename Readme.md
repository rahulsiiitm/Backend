# 🌾 Agricultural Advisory System - Backend

A Flask-based backend API for agricultural management with AI-powered plant disease detection, weather-based farming suggestions, and crop management.

## ✨ Features

- **Plant Disease Detection** - AI model analyzes plant images and identifies diseases
- **Weather-Based Suggestions** - Smart farming recommendations based on current weather
- **Agricultural Chatbot** - AI assistant for farming queries using Gemini
- **Crop Management** - Add, update, and track your crops
- **Daily Suggestions** - Friendly daily farming tips

## 🛠️ Tech Stack

- **Flask** - Web framework
- **TensorFlow/Keras** - Plant disease detection model
- **Google Gemini AI** - Chat assistant and explanations
- **Firebase Firestore** - Database for users, crops, and chat history
- **OpenWeather API** - Real-time weather data

## 📋 Prerequisites

- Python 3.8+
- Firebase project with Firestore enabled
- Google Gemini API key
- OpenWeather API key

## ⚙️ Setup

1. **Clone and install dependencies:**
```bash
pip install flask flask-cors tensorflow google-generativeai firebase-admin requests python-dotenv numpy
```

2. **Environment variables (.env file):**
```
GEMINI_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

3. **Firebase setup:**
   - Place your `firebase-key.json` in the project root
   - Ensure Firestore is enabled

4. **Add your trained model:**
   - Place your `plant_disease_model.keras` in `/models/` folder

## 🚀 API Endpoints

### Chat & Analysis
- `POST /chat` - Chat with AI assistant
- `POST /analyze_image` - Analyze plant disease from image
- `GET /getDailySuggestion` - Get single friendly daily tip

### Crop Management
- `POST /addCrop` - Add new crops
- `GET /getCrops` - Get user's crops
- `PUT /updateCrop` - Update crop details
- `DELETE /deleteCrop` - Delete crop

### Suggestions & Weather
- `GET /getSuggestions` - Get 4 farming suggestions
- `GET /weather` - Get weather data for location

### Chat History
- `GET /getChats` - Get chat history
- `GET /getChat` - Get specific chat
- `DELETE /deleteAllChats` - Delete all chats

## 📝 Usage Examples

### Daily Suggestion
```bash
GET /getDailySuggestion?userId=123&lat=27.1767&lon=78.0081
```
Response:
```json
{
  "success": true,
  "suggestion": {
    "heading": "Sunny day ahead ☀️",
    "body": "Did you water your potato? It's going to be 32°C today!"
  }
}
```

### Add Crop
```bash
POST /addCrop
{
  "user_id": "123",
  "cropData": [{
    "name": "Tomato",
    "type": "vegetable",
    "area": "2",
    "sowedDate": "2024-01-15"
  }]
}
```

### Chat
```bash
POST /chat
{
  "user_id": "123",
  "message": "How do I treat tomato blight?"
}
```

## 🔧 Plant Disease Detection

Supports detection of:
- Tomato diseases (Early blight, Late blight, Leaf mold, etc.)
- Potato diseases (Early blight, Late blight)
- Pepper diseases (Bacterial spot)

## 🌤️ Weather Integration

- Fetches real-time weather data
- Provides 24-hour forecasts
- Integrates weather conditions into farming suggestions

## 🏃‍♂️ Running the Server

```bash
python app.py
```

Server runs on `http://localhost:5000`

## 📊 Database Structure

```
users/
  {userId}/
    crops/
      {cropId}/
        - name, type, area, sowedDate
    chats/
      {chatId}/
        - messages[], createdAt, lastMessage
```

## 🔒 Authentication

- Expects `user_id` parameter in requests
- Integrates with existing login systems
- No user creation handled by this backend

## 🌟 Key Features

- **Smart Suggestions**: Weather-aware farming recommendations
- **AI Chat**: Contextual agricultural advice
- **Disease Detection**: Instant plant health analysis
- **Crop Tracking**: Monitor your crops' growth stages
- **Friendly Interface**: Conversational daily tips

---

🚀 Ready to revolutionize your farming experience!