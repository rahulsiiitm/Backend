import requests
from config import OPENWEATHER_API_KEY
from datetime import datetime

def get_weather_data(lat, lon):
    if not OPENWEATHER_API_KEY:
        return {"success": False, "error": "Weather API key not configured"}
    try:
        # Fetch current weather
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        
        # Fetch 5-day forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        
        if current_response.status_code == 200 and forecast_response.status_code == 200:
            current_data = current_response.json()
            forecast_data_raw = forecast_response.json()
            
            # Format current weather
            current = {
                "temperature": current_data["main"]["temp"],
                "humidity": current_data["main"]["humidity"],
                "description": current_data["weather"][0]["description"],
                "wind_speed": current_data["wind"].get("speed", 0.0),
                "pressure": current_data["main"]["pressure"],
                "feels_like": current_data["main"]["feels_like"],
                "icon": current_data["weather"][0]["icon"]
            }
            
            # Format forecast (extract next few items)
            forecast = []
            for item in forecast_data_raw.get("list", [])[:8]:
                rain = 0
                if "rain" in item and "3h" in item["rain"]:
                    rain = item["rain"]["3h"]
                forecast.append({
                    "date": item["dt_txt"],
                    "temp": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "description": item["weather"][0]["description"],
                    "rain": str(rain)
                })
                
            return {
                "success": True,
                "weather": {
                    "current": current,
                    "forecast": forecast
                }
            }
        return {"success": False, "error": f"Failed to fetch weather: {current_response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
