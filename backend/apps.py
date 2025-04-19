import os
from django.apps import AppConfig

class BackendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend'
    
    def ready(self):
        # Ensure model directory exists
        from django.conf import settings
        os.makedirs(settings.ML_MODEL_CONFIG['MODEL_DIR'], exist_ok=True)