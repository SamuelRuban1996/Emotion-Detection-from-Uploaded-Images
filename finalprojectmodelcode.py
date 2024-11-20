import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Constants
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 7
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def create_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fourth Convolutional Block
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_data_generators(train_dir, test_dir):
    try:
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Using 20% of training data as validation
        )
        
        # Test data generator with only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=BATCH_SIZE,
            shuffle=True,
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=BATCH_SIZE,
            shuffle=True,
            subset='validation'
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator
    
    except Exception as e:
        print(f"Error creating data generators: {str(e)}")
        raise

def train_model(model, train_generator, validation_generator):
    try:
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'best_emotion_model.keras',  # Changed from .h5 to .keras
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )
        
        return history
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

def evaluate_model(model, test_generator):
    try:
        # Get predictions
        test_generator.reset()
        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=EMOTIONS))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Calculate and print test accuracy
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        raise

def plot_training_history(history):
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")
        raise

def main():
    try:
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Get the current working directory
        current_dir = os.getcwd()
        
        # Set your dataset paths
        train_dir = os.path.join(current_dir, 'C:/Samuel/AI ML/Code/fer2013/train')  # Update with your train folder path
        test_dir = os.path.join(current_dir, 'C:/Samuel/AI ML/Code/fer2013/test')    # Update with your test folder path
        
        # Verify directories exist
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise ValueError(f"Test directory not found: {test_dir}")
        
        # Create data generators
        print("Creating data generators...")
        train_generator, validation_generator, test_generator = create_data_generators(train_dir, test_dir)
        
        # Create model
        print("\nCreating model...")
        model = create_model()
        model.summary()
        
        # Train model
        print("\nTraining model...")
        history = train_model(model, train_generator, validation_generator)
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluate_model(model, test_generator)
        
        # Plot training history
        plot_training_history(history)
        
        # Save final model
        model.save('emotion_detection_model.keras')  # Changed from .h5 to .keras
        print("\nModel saved as 'emotion_detection_model.keras'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def predict_emotion(image_path, model):
    """
    Function to predict emotion for a single image
    """
    try:
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            color_mode='grayscale',
            target_size=(IMG_SIZE, IMG_SIZE)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        prediction = model.predict(img_array)
        emotion_idx = np.argmax(prediction[0])
        emotion = EMOTIONS[emotion_idx]
        confidence = prediction[0][emotion_idx]
        
        return emotion, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()