import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Firebase Init
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred)
    except FileNotFoundError:
        print("Warning: serviceAccountKey.json not found.")
db = firestore.client()

# Gemini Init
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
HF_MODEL_API_URL = os.environ.get('HF_MODEL_API_URL')
