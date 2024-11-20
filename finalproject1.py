import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

# Load the emotion detection model
model = tf.keras.models.load_model("emotion_detection_model.keras")
IMG_SIZE = 48
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Constants for image quality checks
MIN_RESOLUTION = 100  # Minimum width or height in pixels
MIN_SHARPNESS_THRESHOLD = 100  # Laplacian variance threshold for blur detection
MIN_FACE_SIZE = 48  # Minimum face size in pixels

def check_image_quality(image_np):
    """
    Check image quality including blur detection and resolution.
    Returns: (is_good_quality, message)
    """
    # Check resolution
    height, width = image_np.shape[:2]
    if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
        return False, f"Image resolution too low. Minimum required: {MIN_RESOLUTION}x{MIN_RESOLUTION}. Got: {width}x{height}"

    # Convert to grayscale for blur detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian variance for blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < MIN_SHARPNESS_THRESHOLD:
        return False, f"Image is too blurry (Sharpness score: {laplacian_var:.2f}). Minimum required: {MIN_SHARPNESS_THRESHOLD}"
    
    return True, "Image quality is acceptable"

def check_face_quality(face_img):
    """
    Check if the detected face is of sufficient quality for emotion detection.
    """
    height, width = face_img.shape[:2]
    
    # Check face size
    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return False, f"Detected face is too small. Minimum size required: {MIN_FACE_SIZE}x{MIN_FACE_SIZE}. Got: {width}x{height}"
    
    # Check face blur specifically
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < MIN_SHARPNESS_THRESHOLD:
        return False, f"Detected face is too blurry (Sharpness score: {laplacian_var:.2f})"
    
    return True, "Face quality is acceptable"

st.title("Emotion Detection App")
st.write("Upload an image, and we'll detect the facial emotion.")

# Mediapipe setup for face detection and landmark extraction
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# File uploader with validation for image files
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Open and validate the uploaded image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Check file size
        if uploaded_file.size > 5000000:  # 5 MB limit
            st.warning("Image file too large! Please upload an image smaller than 5MB.")
        else:
            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Check overall image quality
            is_quality_good, quality_message = check_image_quality(image_np)
            
            if not is_quality_good:
                st.warning(quality_message)
                st.warning("Please upload a clearer, higher quality image for better results.")
            
            # Initialize Mediapipe face detection
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                # Detect faces in the image
                results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

                if not results.detections:
                    st.write("No face detected in the image.")
                else:
                    # Initialize Mediapipe face mesh for landmark extraction
                    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
                        for detection in results.detections:
                            # Get bounding box coordinates
                            bboxC = detection.location_data.relative_bounding_box
                            h, w, _ = image_np.shape
                            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                            
                            # Ensure coordinates are within image bounds
                            x = max(0, x)
                            y = max(0, y)
                            width = min(width, w - x)
                            height = min(height, h - y)
                            
                            face_img = image_np[y:y+height, x:x+width]

                            # Check face quality
                            is_face_quality_good, face_quality_message = check_face_quality(face_img)
                            
                            if not is_face_quality_good:
                                st.warning(face_quality_message)
                                st.warning("The detected face is not of sufficient quality for accurate emotion detection.")
                                continue

                            # Preprocess face for emotion prediction
                            def preprocess_face(face, target_size=(IMG_SIZE, IMG_SIZE)):
                                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                                resized_face = cv2.resize(gray_face, target_size)
                                normalized_face = resized_face / 255.0
                                reshaped_face = np.reshape(normalized_face, (1, IMG_SIZE, IMG_SIZE, 1))
                                return reshaped_face

                            # Process the face
                            processed_face = preprocess_face(face_img)

                            # Predict emotion using the model
                            def predict_emotion(face_img):
                                prediction = model.predict(face_img)
                                emotion_idx = np.argmax(prediction)
                                confidence = prediction[0][emotion_idx]
                                return EMOTIONS[emotion_idx], confidence

                            # Get prediction
                            emotion, confidence = predict_emotion(processed_face)
                            
                            # Only show results if confidence is above threshold
                            if confidence > 0.5:
                                st.write(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
                                
                                # Display face with bounding box and detected emotion
                                cv2.rectangle(image_np, (x, y), (x + width, y + height), (255, 0, 0), 2)
                                st.image(face_img, caption=f"{emotion} ({confidence:.2f})")

                                # Extract and display facial landmarks
                                face_mesh_results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                                if face_mesh_results.multi_face_landmarks:
                                    landmark_img = face_img.copy()
                                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                                        for landmark in face_landmarks.landmark:
                                            x_landmark = int(landmark.x * width)
                                            y_landmark = int(landmark.y * height)
                                            cv2.circle(landmark_img, (x_landmark, y_landmark), 1, (0, 255, 0), -1)
                                    
                                    st.image(landmark_img, caption="Facial Landmarks Detected", use_column_width=True)
                            else:
                                st.warning(f"Low confidence in emotion detection ({confidence:.2f}). Please try with a clearer image.")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.warning("Please try uploading a different image.")

# Add quality control settings in sidebar
with st.sidebar:
    st.subheader("Quality Control Settings")
    MIN_SHARPNESS_THRESHOLD = st.slider("Blur Detection Threshold", 50, 200, 100)
    MIN_RESOLUTION = st.slider("Minimum Resolution (pixels)", 50, 200, 100)
    MIN_FACE_SIZE = st.slider("Minimum Face Size (pixels)", 24, 96, 48)