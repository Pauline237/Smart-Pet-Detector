import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Class for preprocessing images for the cat vs dog classification model.
    
    This class handles various preprocessing steps including:
    - Image loading and resizing
    - Normalization
    - Augmentation (for training)
    - Batch generation
    """
    
    def __init__(self, target_size=(224, 224), rescale=1./255):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            target_size (tuple): Target image dimensions (height, width)
            rescale (float): Rescaling factor for pixel values
        """
        self.target_size = target_size
        self.rescale = rescale
        logger.info(f"ImagePreprocessor initialized with target_size={target_size}")
    
    def load_and_preprocess_image(self, image_path, grayscale=False):
        """
        Load and preprocess a single image from file path.
        
        Args:
            image_path (str): Path to the image file
            grayscale (bool): Whether to convert image to grayscale
            
        Returns:
            numpy.ndarray: Preprocessed image as numpy array
        """
        try:
            # Load image
            if grayscale:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for model compatibility
            else:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Convert to array and rescale
            image = img_to_array(image) * self.rescale
            
            logger.debug(f"Successfully preprocessed image: {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def preprocess_pil_image(self, pil_image):
        """
        Preprocess an image from PIL Image object.
        
        Args:
            pil_image (PIL.Image): PIL Image object
            
        Returns:
            numpy.ndarray: Preprocessed image as numpy array
        """
        try:
            # Convert PIL image to numpy array
            image = np.array(pil_image.convert('RGB'))
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Convert to array and rescale
            image = img_to_array(image) * self.rescale
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            logger.debug("Successfully preprocessed PIL image")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing PIL image: {str(e)}")
            raise
    
    def preprocess_uploaded_file(self, file_object):
        """
        Preprocess an uploaded file object.
        
        Args:
            file_object: Django uploaded file object
            
        Returns:
            numpy.ndarray: Preprocessed image as numpy array
        """
        try:
            # Open image with PIL
            pil_image = Image.open(file_object)
            
            # Use the PIL preprocessing method
            return self.preprocess_pil_image(pil_image)
            
        except Exception as e:
            logger.error(f"Error preprocessing uploaded file: {str(e)}")
            raise
    
    def preprocess_directory(self, directory_path, batch_size=32):
        """
        Generator to load and preprocess images from a directory in batches.
        
        Args:
            directory_path (str): Path to the directory containing images
            batch_size (int): Number of images to process in each batch
            
        Yields:
            tuple: (batch_images, batch_paths) where batch_images is a numpy array 
                  of shape (batch_size, *target_size, 3) and batch_paths are the 
                  corresponding file paths
        """
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            logger.warning(f"No image files found in {directory_path}")
            return
        
        logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        for i in range(0, len(image_files), batch_size):
            batch_paths = [os.path.join(directory_path, f) for f in image_files[i:i+batch_size]]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = self.load_and_preprocess_image(path)
                    batch_images.append(img)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
                    continue
            
            if batch_images:
                batch_images = np.array(batch_images)
                yield batch_images, batch_paths
            else:
                logger.warning(f"No valid images in batch starting at index {i}")


# Create a utility function for easy access
def get_preprocessor(target_size=(224, 224)):
    """Factory function to create an ImagePreprocessor with default settings."""
    return ImagePreprocessor(target_size=target_size)