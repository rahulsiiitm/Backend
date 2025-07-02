# AgriHive - Agricultural Advisory System 🌱

A Flask-based hobby project exploring AI-powered agriculture! This backend API combines plant disease detection using my custom-trained model, weather-based farming suggestions, and comprehensive crop management. Built as a personal learning project to explore agricultural technology and AI integration.

> **Personal Learning Project** - Combining my interests of AI with real-time projects

## Features

- **Plant Disease Detection**
- **Weather-Based Suggestions**
- **Agricultural Chatbot**
- **Crop Management**
- **Daily Suggestions**
- **Chat History**

## Technology Stack

- **Flask** - Web framework with CORS support
- **Custom AI Model** - 13-class plant disease detection model (hosted on Hugging Face)
- **Google Gemini AI** - Chat assistant and agricultural explanations
- **Firebase Firestore** - Database for users, crops, and chat history
- **OpenWeather API** - Real-time weather data and forecasts
- **Render** - Cloud hosting platform


## Prerequisites

- Python 3.8+
- Firebase project with Firestore enabled
- Google Gemini API key
- OpenWeather API key
- Custom Hugging Face model endpoint

## Installation & Setup

### Local Development

1. **Clone and install:**
```bash
git clone <repository-url>
cd agricultural-advisory-system
pip install -r requirements.txt
```

2. **Environment variables (.env):**
```env
GEMINI_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
HF_MODEL_API_URL=your_model_api_url
FIREBASE_KEY=your_firebase_service_account_json_string
```

3. **Firebase setup:**
   - Create Firebase project with Firestore enabled
   - Generate service account key
   - Add to environment variables

4. **Run the application:**
```bash
python app.py  # Runs on http://localhost:5000
```

### Production Deployment

1. Connect GitHub repository to Render
2. Configure environment variables in Render dashboard
3. Auto-deploy on push to main branch

## API Endpoints

### Base URLs
- **Local**: `http://localhost:5000`
- **Production**: `https://your-app-name.onrender.com`

### Chat & AI Analysis
- `POST /chat` - Interactive chat with AI assistant
- `POST /analyze_image` - Plant disease detection using custom 13-class model

### Crop Management
- `POST /addCrop` - Add new crops to collection
- `GET /getCrops` - Retrieve user's crop data
- `PUT /updateCrop` - Update existing crop information
- `DELETE /deleteCrop` - Remove crops from collection

### Weather & Suggestions
- `GET /getSuggestions` - Generate 4 weather-based farming suggestions
- `GET /getDailySuggestion` - Get personalized daily farming tip
- `GET /weather` - Fetch current weather and forecast data

### Chat Management
- `GET /getChats` - Retrieve chat history
- `GET /getChat` - Get specific chat conversation
- `DELETE /deleteAllChats` - Clear all chat history

### User Profile
- `GET /get_farmer_profile` - Retrieve user profile information
- `POST /update_farmer_profile` - Update user profile data

### System Health
- `GET /` - API information and available endpoints
- `GET /health` - System health check

## Usage Examples

### Adding Crops
```bash
POST /addCrop
Content-Type: application/json

{
  "user_id": "user123",
  "cropData": [
    {
      "name": "Tomato",
      "type": "Vegetable",
      "area": "2 acres",
      "sowedDate": "2024-01-15"
    }
  ]
}
```

### Disease Detection with Custom Model
```bash
POST /analyze_image
Content-Type: multipart/form-data

FormData:
- image: [plant_image_file]
- user_id: user123
- chat_id: optional_existing_chat_id
```

### Getting Weather-Based Suggestions
```bash
GET /getSuggestions?userId=user123&lat=27.1767&lon=78.0081
```

### AI Chat Interaction
```bash
POST /chat
Content-Type: application/json

{
  "user_id": "user123",
  "message": "How should I treat tomato blight?",
  "chat_id": "optional_existing_chat_id"
}
```

## My Custom Plant Disease Detection Model

### Model Details
- **13 Disease Classes** - Trained to identify common plant diseases
- **Custom Dataset** - Curated specifically for agricultural conditions
- **Hugging Face Integration** - Hosted for easy API access
- **Real-time Analysis** - Quick disease identification from uploaded plant photos

### Supported Disease Classes
The model can detect 13 different plant diseases across various crops, providing:
- Disease identification
- Confidence scores
- Treatment recommendations via AI chat integration

## Response Formats

### Successful Crop Addition
```json
{
  "message": "Crop(s) added successfully",
  "userId": "user123",
  "cropsAdded": [
    {
      "cropId": "generated_uuid",
      "data": {
        "name": "Tomato",
        "type": "Vegetable",
        "area": "2 acres",
        "sowedDate": "2024-01-15",
        "timestamp": "2024-01-15T10:30:00"
      }
    }
  ]
}
```

### Disease Detection Response
```json
{
  "success": true,
  "disease_detected": "Tomato Late Blight",
  "confidence": 0.89,
  "chat_response": "AI-generated treatment advice",
  "chat_id": "generated_chat_id"
}
```

### Daily Suggestion Response
```json
{
  "success": true,
  "suggestion": {
    "heading": "Morning Farm Check",
    "body": "Check your tomato plants for early signs of blight - humidity is 75% today which increases disease risk."
  }
}
```

## Database Structure

```
users/
  {userId}/
    crops/
      {cropId}
        - name: string
        - type: string
        - area: string
        - sowedDate: string
        - timestamp: datetime
    chats/
      {chatId}
        - messages: array
        - createdAt: datetime
        - lastMessage: string
        - updatedAt: datetime
    profile/
      info
        - name: string
        - phone: string
        - location: string
        - language: string
        - profilePhoto: string
```

## Dependencies

```txt
flask==2.3.3
flask-cors==4.0.0
google-generativeai==0.3.0
firebase-admin==6.2.0
requests==2.31.0
python-dotenv==1.0.0
gunicorn==21.2.0
```

## Configuration

### Environment Variables
```env
# AI Services
GEMINI_API_KEY=your_gemini_api_key_here
HF_MODEL_API_URL=https://your-custom-model-endpoint.com

# Weather Service
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Database
FIREBASE_KEY={"type":"service_account","project_id":"..."}
```

### Default Settings
- Default location: Agra, Uttar Pradesh (27.1767, 78.0081)
- Request timeout: 30 seconds for external APIs
- Auto-create Firebase collections

## Error Handling

Comprehensive error handling for:
- Invalid user IDs and missing parameters
- External API failures (Gemini, OpenWeather, Custom Model)
- Database connection issues
- Image processing errors
- Non-plant image detection

## What I Learned

### Technical Skills
- Flask API development and deployment
- AI model integration and API design
- Cloud database management with Firebase
- Weather API integration and data processing
- Image processing and computer vision basics
- Production deployment on cloud platforms

### Agricultural Technology
- Plant disease identification challenges
- Weather impact on farming decisions
- Digital crop management systems
- AI-powered agricultural assistance

## Future Enhancements

- Mobile app frontend
- Additional disease classes for the model
- Crop yield prediction features
- Multi-language support
- Advanced analytics dashboard
- Community features for hobby farmers

## Deployment Notes

### Render Configuration
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`
- Environment: Python 3.8+
- Auto-deploy: Enabled from main branch

### Production Considerations
- Uses Gunicorn WSGI server
- Health checks via `/health` endpoint
- Environment variables securely stored
- CORS configured for frontend integration

## Personal Project Notes

**This is a hobby project built for learning!**

- Exploring agricultural AI applications
- Custom disease detection model training
- Learning cloud deployment and API design
- Combining interests in farming and technology
- Not intended for commercial use

---

*A personal exploration of AI in agriculture* 🌾💻
