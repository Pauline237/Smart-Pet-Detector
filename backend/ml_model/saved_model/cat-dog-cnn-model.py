import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def build_model(input_shape=(150, 150, 3)):
    """
    Build a CNN model for cat vs dog classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),  # Reduce overfitting
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators(train_dir, validation_dir, test_dir=None, batch_size=32, img_size=(150, 150)):
    """
    Create data generators for training, validation, and optionally test data
    
    Args:
        train_dir: Directory with training data
        validation_dir: Directory with validation data
        test_dir: Directory with test data (optional)
        batch_size: Number of images per batch
        img_size: Target size for images (height, width)
        
    Returns:
        Tuple of data generators (train_generator, validation_generator, test_generator)
    """
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test data
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'  # Binary labels
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    # Create test generator if test directory is provided
    test_generator = None
    if test_dir:
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
    
    return train_generator, validation_generator, test_generator

def train_model(model, train_generator, validation_generator, epochs=20, 
                callbacks=None, model_save_path='saved_model'):
    """
    Train the model and save it
    
    Args:
        model: Compiled Keras model
        train_generator: Generator for training data
        validation_generator: Generator for validation data
        epochs: Number of training epochs
        callbacks: List of callbacks to use during training
        model_save_path: Directory to save the trained model
        
    Returns:
        Training history object
    """
    if callbacks is None:
        # Create default callbacks
        os.makedirs(model_save_path, exist_ok=True)
        
        # Checkpoint to save best model
        checkpoint_path = os.path.join(model_save_path, 'best_model.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Reduce learning rate when a metric has stopped improving
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )
    
    # Save final model in TensorFlow SavedModel format
    saved_model_path = os.path.join(model_save_path, 'final_model')
    model.save(saved_model_path)
    
    return history

def plot_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: History object from model training
        save_path: Path to save the plot image (optional)
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def create_and_train_model(train_dir, validation_dir, test_dir=None, 
                           img_size=(150, 150), batch_size=32, epochs=20,
                           model_save_path='saved_model'):
    """
    Create and train a cat vs dog classification model
    
    Args:
        train_dir: Directory with training data
        validation_dir: Directory with validation data
        test_dir: Directory with test data (optional)
        img_size: Target size for images (height, width)
        batch_size: Number of images per batch
        epochs: Number of training epochs
        model_save_path: Directory to save the trained model
        
    Returns:
        Tuple of (model, history)
    """
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, validation_dir, test_dir, batch_size, img_size
    )
    
    # Build model
    model = build_model(input_shape=(*img_size, 3))
    
    # Print model summary
    model.summary()
    
    # Train model
    history = train_model(
        model, train_generator, validation_generator, 
        epochs=epochs, model_save_path=model_save_path
    )
    
    # Plot training history
    plot_history(history, save_path=os.path.join(model_save_path, 'training_history.png'))
    
    # Evaluate on test set if available
    if test_generator:
        test_loss, test_acc = model.evaluate(test_generator)
        print(f'Test accuracy: {test_acc:.4f}')
    
    return model, history

def predict_image(model, image_path, img_size=(150, 150)):
    """
    Predict cat or dog for a single image
    
    Args:
        model: Trained model
        image_path: Path to the image file
        img_size: Image size expected by the model
        
    Returns:
        Tuple of (prediction_label, confidence)
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Determine label and confidence
    if prediction >= 0.5:
        label = 'dog'
        confidence = float(prediction)
    else:
        label = 'cat'
        confidence = float(1 - prediction)
    
    return label, confidence

if __name__ == "__main__":
    # Example usage:
    # Replace these with your actual data directories
    base_dir = '/path/to/cats_and_dogs'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f'pet_classifier/ml_model/saved_model/{timestamp}'
    
    # Create and train the model
    model, history = create_and_train_model(
        train_dir=train_dir,
        validation_dir=validation_dir,
        test_dir=test_dir,
        epochs=20,
        model_save_path=model_save_path
    )
    
    print(f"Model saved to {model_save_path}")