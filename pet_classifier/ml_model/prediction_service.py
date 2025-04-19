import os
import time
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from django.conf import settings

# Import model and preprocessing modules
from .model import PetClassifierModel
from .data_preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for making predictions using the trained pet classifier model.
    
    This class handles:
    - Loading the trained model
    - Processing input images
    - Making and returning predictions
    - Managing model versioning
    """
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the prediction service."""
        # Skip initialization if already initialized
        if self._initialized:
            return
            
        self.model = None
        self.preprocessor = None
        self.model_version = None
        self.model_path = None
        self.target_size = (224, 224)
        
        # Initialize the service
        self._initialize()
        self._initialized = True
    
    def _initialize(self):
        """Initialize the prediction service by loading model and preprocessor."""
        try:
            # Determine model path
            if hasattr(settings, 'ML_MODEL_PATH'):
                model_dir = settings.ML_MODEL_PATH
            else:
                # Default path if settings not available
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(current_dir, 'saved_model')
            
            # Find the latest model file
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
                if model_files:
                                       # Sort by modification time (latest first)
                    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
                    self.model_path = os.path.join(model_dir, model_files[0])
                    self.model_version = os.path.splitext(model_files[0])[0]
                else:
                    logger.warning(f"No model files found in {model_dir}")
            else:
                logger.warning(f"Model directory not found: {model_dir}")
            
            # Initialize preprocessor
            self.preprocessor = ImagePreprocessor(target_size=self.target_size)
            
            # Load model if path exists
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = PetClassifierModel(model_path=self.model_path)
                logger.info(f"Model loaded successfully. Version: {self.model_version}")
            else:
                logger.error("No valid model path available for loading")
        
        except Exception as e:
            logger.error(f"Error initializing PredictionService: {str(e)}")
            raise
    
    def predict_image_file(self, file_path):
        """
        Make prediction for an image file.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            dict: Prediction result containing:
                - class_name (str): 'cat' or 'dog'
                - confidence (float): Confidence score (0.0 to 1.0)
                - processing_time (float): Time taken for prediction in seconds
                - model_version (str): Version of the model used
        """
        try:
            start_time = time.time()
            
            # Preprocess the image
            image = self.preprocessor.load_and_preprocess_image(file_path)
            
            # Make prediction
            class_id, confidence = self.model.predict(image)
            
            # Determine class name
            class_name = 'dog' if class_id == 1 else 'cat'
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.info(f"Prediction: {class_name} (confidence: {confidence:.2f}) in {processing_time:.2f}s")
            
            return {
                'class_name': class_name,
                'confidence': float(confidence),
                'processing_time': processing_time,
                'model_version': self.model_version
            }
            
        except Exception as e:
            logger.error(f"Error predicting image {file_path}: {str(e)}")
            raise
    
    def predict_uploaded_file(self, uploaded_file):
        """
        Make prediction for a Django uploaded file.
        
        Args:
            uploaded_file: Django InMemoryUploadedFile or TemporaryUploadedFile
            
        Returns:
            dict: Prediction result (same format as predict_image_file)
        """
        try:
            start_time = time.time()
            
            # Preprocess the uploaded file
            image = self.preprocessor.preprocess_uploaded_file(uploaded_file)
            
            # Make prediction
            class_id, confidence = self.model.predict(image)
            
            # Determine class name
            class_name = 'dog' if class_id == 1 else 'cat'
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.info(f"Prediction: {class_name} (confidence: {confidence:.2f}) in {processing_time:.2f}s")
            
            return {
                'class_name': class_name,
                'confidence': float(confidence),
                'processing_time': processing_time,
                'model_version': self.model_version
            }
            
        except Exception as e:
            logger.error(f"Error predicting uploaded file: {str(e)}")
            raise
    
    def predict_pil_image(self, pil_image):
        """
        Make prediction for a PIL Image object.
        
        Args:
            pil_image: PIL.Image object
            
        Returns:
            dict: Prediction result (same format as predict_image_file)
        """
        try:
            start_time = time.time()
            
            # Preprocess the PIL image
            image = self.preprocessor.preprocess_pil_image(pil_image)
            
            # Make prediction
            class_id, confidence = self.model.predict(image)
            
            # Determine class name
            class_name = 'dog' if class_id == 1 else 'cat'
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.info(f"Prediction: {class_name} (confidence: {confidence:.2f}) in {processing_time:.2f}s")
            
            return {
                'class_name': class_name,
                'confidence': float(confidence),
                'processing_time': processing_time,
                'model_version': self.model_version
            }
            
        except Exception as e:
            logger.error(f"Error predicting PIL image: {str(e)}")
            raise
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information containing:
                - version (str): Model version
                - input_shape (tuple): Expected input shape
                - last_modified (str): Last modified timestamp
        """
        if not self.model_path or not os.path.exists(self.model_path):
            return None
            
        return {
            'version': self.model_version,
            'input_shape': self.target_size + (3,),  # Add channels dimension
            'last_modified': time.ctime(os.path.getmtime(self.model_path))
        }


def get_prediction_service():
    """Factory function to get the PredictionService singleton instance."""
    return PredictionService()