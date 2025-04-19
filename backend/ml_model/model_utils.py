import os
import requests
from django.conf import settings

def download_default_model():
    model_url = "https://github.com/example/pet-classifier-models/releases/latest/download/pet_classifier_model.h5"
    model_path = os.path.join(settings.ML_MODEL_CONFIG['MODEL_DIR'], 'pet_classifier_model.h5')
    
    if not os.path.exists(model_path):
        os.makedirs(settings.ML_MODEL_CONFIG['MODEL_DIR'], exist_ok=True)
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"Failed to download default model: {str(e)}")
            return False
    return True