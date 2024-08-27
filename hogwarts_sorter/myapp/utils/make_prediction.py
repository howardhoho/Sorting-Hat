import os
import numpy as np
import tensorflow as tf
import joblib
import cv2
import base64
from .face_emo import EmoExtractor
from .face_feature import FeatureExtractor
from flask import jsonify

# Define the directory path
model_dir = os.path.join(os.path.dirname(__file__), 'model')

cnn_model_path = os.path.join(model_dir, 'new_hogwarts_cnn_model.h5')
rf_model_path = os.path.join(model_dir, 'aug_random_forest_house_classifier.pkl')


# Load the pre-trained models and class labels once, at the start
cnn_model = tf.keras.models.load_model(cnn_model_path)
rf_model = joblib.load(rf_model_path)
class_labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

def process_image(img_cv):
    print("Started processing the image.")

    try:
        # Initialize Emo and Feature Extractors
        emo_extractor = EmoExtractor(img_cv, "model")
        feature_extractor = FeatureExtractor(img_cv, "model")

        print("Initialized extractors")

        # Process image and extract landmarks
        resized_img = emo_extractor.process_img()
        landmark_img = feature_extractor.face_landmark()

        if resized_img is None:
            return {'error': 'Failed to process image'}, 400

        print("Processed image")

        # Get Emotion and Feature Predictions
        pred_emo = emo_extractor.predict(resized_img)
        pred_feature = feature_extractor.face_feature()

        print("Predictions extracted")

        # Combine features
        x_face_info = list(pred_emo) + pred_feature
        x_face_info = np.array(x_face_info).reshape(1, -1)

        # Random Forest Prediction
        rf_probs = rf_model.predict_proba(x_face_info)[0]  # Ensure it's a 1D array

        # CNN Prediction
        img = cv2.resize(img_cv, (150, 150))
        img_array = img / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        cnn_probs = cnn_model.predict(img_array)[0]  # Also a 1D array

        print("Predictions complete")

        # Combine CNN and Random Forest predictions with weighting
        final_pred = cnn_probs + rf_probs * [1.5, 5, 1, 2] + np.array([0.4, 0.1, 0, 0.42])
        predicted_class = np.argmax(final_pred)  # Get the index of the highest probability
        predicted_label = class_labels[predicted_class]

        print(f"Predicted label: {predicted_label}")

        # Convert images to base64 for JSON response
        _, buffer = cv2.imencode('.jpg', resized_img)
        base64_resized_img = base64.b64encode(buffer).decode('utf-8')

        _, l_buffer = cv2.imencode('.jpg', landmark_img)
        base64_landmark_img = base64.b64encode(l_buffer).decode('utf-8')

        return {
        'prediction': predicted_label,
        'resized_img': base64_resized_img,
        'landmarked_img': base64_landmark_img
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {'error': str(e)}, 500


