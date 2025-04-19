import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PetClassifierModel:
    """
    CNN model for classifying images as cats or dogs.
    
    This class provides methods for:
    - Creating a CNN model architecture
    - Loading pretrained weights
    - Training the model
    - Making predictions
    """
    
    def __init__(self, model_path=None, input_shape=(224, 224, 3)):
        """
        Initialize the PetClassifierModel.
        
        Args:
            model_path (str, optional): Path to saved model weights
            input_shape (tuple): Input image shape (height, width, channels)
        """
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        
        # Build the model
        self._build_model()
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        logger.info(f"PetClassifierModel initialized with input_shape={input_shape}")
    
    def _build_model(self):
        """Build the CNN model architecture."""
        logger.info("Building model architecture...")
        
        # Use a pre-trained MobileNetV2 as the base model
        # MobileNetV2 is a good balance between accuracy and speed for mobile/web applications
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model to prevent training its weights
        base_model.trainable = False
        
        # Create the model architecture
        model = models.Sequential([
            # Base model
            base_model,
            
            # Add custom classification layers
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification (cat vs dog)
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Model built successfully: {model.summary()}")
        return model
    
    def unfreeze_top_layers(self, num_layers=20):
        """
        Unfreeze the top layers of the base model for fine-tuning.
        
        Args:
            num_layers (int): Number of top layers to unfreeze
        """
        if self.model is None:
            logger.error("Model not built yet")
            return
        
        # Get the base model (first layer)
        base_model = self.model.layers[0]
        
        # Unfreeze the top num_layers layers
        for layer in base_model.layers[-(num_layers):]:
            layer.trainable = True
        
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Unfroze top {num_layers} layers for fine-tuning")
    
    def train(self, train_data, validation_data, epochs=20, batch_size=32, callbacks=None):
        """
        Train the model.
        
        Args:
            train_data: Training data as tf.data.Dataset or tuple (x_train, y_train)
            validation_data: Validation data as tf.data.Dataset or tuple (x_val, y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): List of Keras callbacks for training
            
        Returns:
            History object from model.fit()
        """
        if self.model is None:
            logger.error("Model not built yet")
            return
        
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        # If data is provided as tuples, use direct fit method
        if isinstance(train_data, tuple):
            history = self.model.fit(
                train_data[0], train_data[1],
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # If data is provided as tf.data.Dataset
            history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info("Training completed")
        return history
    
    def _get_default_callbacks(self):
        """Create default callbacks for training."""
        if not self.model_path:
            model_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(model_dir, 'saved_model', 'pet_classifier_model.h5')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model_path = model_path
            
        callbacks = [
            # Save the best model based on validation accuracy
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Stop training if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate if validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        return callbacks
    
    def predict(self, image):
        """
        Make prediction for a single preprocessed image.
        
        Args:
            image (numpy.ndarray): Preprocessed image array
            
        Returns:
            tuple: (class_id, confidence) where class_id is 0 for cat, 1 for dog
        """
        if self.model is None:
            logger.error("Model not built yet")
            return None
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image)
        confidence = float(prediction[0][0])
        
        # Class ID: 0 for cat, 1 for dog
        class_id = 1 if confidence >= 0.5 else 0
        
        logger.debug(f"Prediction: class_id={class_id}, confidence={confidence:.4f}")
        return class_id, confidence
    
    def save_model(self, path=None):
        """
        Save the model to disk.
        
        Args:
            path (str, optional): Path to save the model. 
                                  If None, use the instance's model_path.
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        if path is None:
            path = self.model_path
        
        if path is None:
            logger.error("No path specified to save model")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """
        Load model weights from disk.
        
        Args:
            path (str, optional): Path to the saved model.
                                  If None, use the instance's model_path.
        """
        if path is None:
            path = self.model_path
        
        if path is None or not os.path.exists(path):
            logger.error(f"Model path does not exist: {path}")
            return
        
        # Load the model
        try:
            self.model = models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
    
    def export_to_tf_lite(self, output_path):
        """
        Export the model to TensorFlow Lite format.
        
        Args:
            output_path (str): Path to save the TF Lite model
        """
        if self.model is None:
            logger.error("No model to export")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to TF Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"Model exported to TF Lite format at {output_path}")
    
    def get_model_summary(self):
        """Get a string representation of the model architecture."""
        if self.model is None:
            return "Model not built yet"
        
        # Create a string buffer to capture summary
        from io import StringIO
        buffer = StringIO()
        
        # Save summary to buffer
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        
        # Return the summary as string
        return buffer.getvalue()


def get_model(model_path=None, input_shape=(224, 224, 3)):
    """Factory function to create a PetClassifierModel with default settings."""
    return PetClassifierModel(model_path=model_path, input_shape=input_shape)