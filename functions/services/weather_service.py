import requests
from config import OPENWEATHER_API_KEY

def get_weather_data(lat, lon):
    if not OPENWEATHER_API_KEY:
        return {"error": "Weather API key not configured"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "icon": data["weather"][0]["icon"],
                "city": data.get("name", "Unknown Location")
            }
        return {"error": f"Failed to fetch weather: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
