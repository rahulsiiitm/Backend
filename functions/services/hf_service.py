import requests
from config import HF_MODEL_API_URL

def call_hf_model_api(image_data, is_file=True):
    try:
        if not HF_MODEL_API_URL:
            raise Exception("Hugging Face model API URL not configured")
        
        if is_file:
            files = {'image': image_data}
            response = requests.post(f"{HF_MODEL_API_URL}/predict", files=files, timeout=30)
        else:
            headers = {'Content-Type': 'application/json'}
            data = {'image': image_data}
            response = requests.post(f"{HF_MODEL_API_URL}/predict", json=data, headers=headers, timeout=30)
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Model API error: {str(e)}")
