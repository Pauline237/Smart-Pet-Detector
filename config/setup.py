import os
import warnings
from django.conf import settings

def configure_tensorflow():
    """Configure TensorFlow environment variables"""
    # Suppress most TF logs (0 = all, 1 = no info, 2 = no warnings, 3 = no errors)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    
    # Disable GPU if not available
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            tf.config.set_visible_devices([], 'GPU')
    except ImportError:
        pass

    # Suppress TensorRT warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

def verify_staticfiles():
    """Ensure staticfiles directories exist"""
    os.makedirs(settings.STATICFILES_DIRS[0], exist_ok=True)
    os.makedirs(settings.STATIC_ROOT, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['css', 'js', 'images']:
        os.makedirs(os.path.join(settings.STATICFILES_DIRS[0], subdir), exist_ok=True)

def setup_environment():
    """Run all configuration checks"""
    configure_tensorflow()
    verify_staticfiles()