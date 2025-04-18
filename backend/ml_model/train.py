#!/usr/bin/env python3
"""
Training script for the Pet Classifier model.

This script handles:
- Loading and preprocessing training data
- Building and training the CNN model 
- Evaluating model performance
- Saving the trained model
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Add the project root to sys.path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(project_dir)

# Import the model and preprocessing modules
from pet_classifier.ml_model.model import PetClassifierModel
from pet_classifier.ml_model.data_preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(script_dir, 'training.log'))
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the Pet Classifier model')
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing cat and dog subdirectories with training images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(script_dir, 'saved_model'),
        help='Directory to save the trained model'
    )
    
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Size to resize images to (square)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--fine_tune',
        action='store_true',
        help='Fine-tune the pre-trained base model'
    )
    
    parser.add_argument(
        '--fine_tune_epochs',
        type=int,
        default=10,
        help='Number of fine-tuning epochs'
    )
        
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the model after training'
    )
    
    return parser.parse_args()


def prepare_data_generators(data_dir, img_size, batch_size):
    """
    Prepare data generators for training, validation, and testing.
    
    Args:
        data_dir (str): Directory containing the dataset
        img_size (int): Size to resize images to
        batch_size (int): Batch size for generators
        
    Returns:
        tuple: (train_generator, validation_generator, test_generator)
    """
    logger.info(f"Preparing data generators from {data_dir}")
    
    # Define data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescale for validation/testing
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Training generator with augmentation
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',  # binary labels for binary_crossentropy loss
        subset='training'
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # Test generator (assumes test data is in a separate directory)
    test_dir = os.path.join(os.path.dirname(data_dir), 'test')
    if os.path.exists(test_dir):
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
    else:
        logger.warning(f"Test directory not found at {test_dir}, using validation data for testing")
        test_generator = validation_generator
    
    logger.info(f"Found {train_generator.samples} training images")
    logger.info(f"Found {validation_generator.samples} validation images")
    logger.info(f"Found {test_generator.samples} test images")
    
    return train_generator, validation_generator, test_generator


def train_model(args):
    """
    Train the pet classifier model.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (model, history) - trained model and training history
    """
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate a timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f'pet_classifier_model_{timestamp}.h5')
    
    # Prepare data generators
    train_generator, validation_generator, test_generator = prepare_data_generators(
        args.data_dir, args.img_size, args.batch_size
    )
    
    # Create model
    input_shape = (args.img_size, args.img_size, 3)
    model = PetClassifierModel(input_shape=input_shape)
    
    # Log model summary
    logger.info("Model architecture:")
    logger.info(model.get_model_summary())
    
    # Train initial model with frozen base
    logger.info("Starting initial training phase...")
    history = model.train(
        train_generator,
        validation_generator,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Fine-tune if requested
    if args.fine_tune:
        logger.info("Starting fine-tuning phase...")
        model.unfreeze_top_layers(num_layers=20)
        
        # Continue training with unfrozen layers
        fine_tune_history = model.train(
            train_generator,
            validation_generator,
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size // 2  # Reduce batch size for fine-tuning
        )
        
        # Merge histories
        for key in fine_tune_history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Save the model
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save TF Lite version for mobile
    tflite_path = os.path.join(args.output_dir, f'pet_classifier_model_{timestamp}.tflite')
    model.export_to_tf_lite(tflite_path)
    logger.info(f"TF Lite model saved to {tflite_path}")
    
    # Plot training history
    plot_training_history(history, args.output_dir, timestamp)
    
    # Validate if requested
    if args.validate:
        evaluate_model(model, test_generator, args.output_dir, timestamp)
    
    return model, history


def plot_training_history(history, output_dir, timestamp):
    """
    Plot and save training history.
    
    Args:
        history: Training history object from model.fit()
        output_dir (str): Directory to save plots
        timestamp (str): Timestamp for filename
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'training_history_{timestamp}.png')
    plt.savefig(plot_path)
    logger.info(f"Training history plot saved to {plot_path}")


def evaluate_model(model, test_generator, output_dir, timestamp):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_generator: Test data generator
        output_dir (str): Directory to save evaluation results
        timestamp (str): Timestamp for filename
    """
    # Create reports directory
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions
    logger.info("Evaluating model on test data...")
    steps = np.ceil(test_generator.samples / test_generator.batch_size)
    
    # Get predictions and true labels
    predictions = []
    true_labels = []
    
    for i in range(int(steps)):
        x_batch, y_batch = next(test_generator)
        batch_preds = model.model.predict(x_batch)
        predictions.extend(batch_preds)
        true_labels.extend(y_batch)
        
        if (i+1) * test_generator.batch_size >= test_generator.samples:
            break
    
    # Convert to binary predictions
    binary_preds = [1 if pred[0] >= 0.5 else 0 for pred in predictions]
    binary_true = [1 if label >= 0.5 else 0 for label in true_labels]
    
    # Calculate metrics
    cm = confusion_matrix(binary_true, binary_preds)
    report = classification_report(binary_true, binary_preds, 
                                  target_names=['Cat', 'Dog'])
    
    # Save reports
    report_path = os.path.join(reports_dir, f'evaluation_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
    
    logger.info(f"Evaluation report saved to {report_path}")
    logger.info("\nClassification Report:\n" + report)


def main():
    """Main function to train the model."""
    # Parse arguments
    args = parse_arguments()
    
    # Log GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Training with {len(gpus)} GPU(s): {gpus}")
    else:
        logger.info("Training with CPU")
    
    # Train the model
    logger.info("Starting training process...")
    model, history = train_model(args)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()