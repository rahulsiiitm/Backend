import json
import google.generativeai as genai
from config import GEMINI_API_KEY

def generate_farming_suggestions_with_gemini(crops, weather_data):
    if not GEMINI_API_KEY:
        return [{"category": "system", "crop": "All", "priority": "Low", "text": "AI suggestions unavailable"}] * 4
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        crop_text = "\n".join([f"- {c.get('name')} ({c.get('type')})" for c in crops])
        weather_text = f"{weather_data.get('temp')}C, {weather_data.get('description')}" if not weather_data.get('error') else "Unknown"
        
        prompt = f"""Act as an expert agricultural advisor. Farmer is growing:
{crop_text}
Weather: {weather_text}
Provide exactly 4 actionable farming suggestions.
Format as a JSON array of exactly 4 objects. Each object must have these keys:
- category (e.g., 'irrigation', 'fertilizer', 'disease', 'general')
- crop (the name of the crop it applies to, or 'All')
- priority ('High', 'Medium', 'Low')
- text (The actual suggestion text)
Return ONLY valid JSON."""
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text: text = text.split('```')[1].strip()
        suggestions = json.loads(text)
        
        while len(suggestions) < 4:
            suggestions.append({"category": "general", "crop": "All", "priority": "Low", "text": "Continue regular monitoring."})
        return suggestions[:4]
    except Exception as e:
        print(f"Error in Gemini: {e}")
        return [{"category": "error", "crop": "All", "priority": "Low", "text": "Unable to generate suggestions."}] * 4

def generate_daily_suggestion_with_gemini(weather_data):
    if not GEMINI_API_KEY:
        return {"title": "Tip", "content": "Monitor crops.", "type": "general"}
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        weather_text = f"{weather_data.get('temp')}C, {weather_data.get('description')}" if not weather_data.get('error') else "Unknown"
        prompt = f"Act as an agricultural advisor. Weather: {weather_text}\nProvide one daily farming suggestion. Format strictly as JSON with keys: title, content, type."
        response = model.generate_content(prompt)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text: text = text.split('```')[1].strip()
        return json.loads(text)
    except:
        return {"title": "Tip", "content": "Ensure proper irrigation.", "type": "general"}
