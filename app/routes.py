from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from scipy.optimize import minimize
from tensorflow.keras.preprocessing import image
import base64
from PIL import Image
from face_emo import EmoExtractor
from face_feature import FeatureExtractor
import os
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
cnn_model = tf.keras.models.load_model('model/new_hogwarts_cnn_model.h5')
rf_model = joblib.load('model/aug_random_forest_house_classifier.pkl')

# Define the class labels
class_labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)
    
    try:
        # Emo and Feature Probability
        emo_extractor = EmoExtractor(image_path)
        feature_extractor = FeatureExtractor(image_path)

        resized_img = emo_extractor.process_img()
        landmark_img = feature_extractor.face_landmark()

        if resized_img is None:
            return jsonify({'error': 'Failed to process image'}), 400

        pred_emo = emo_extractor.predict(resized_img)
        pred_feature = feature_extractor.face_feature()

        # Get probabilities from Random Forest
        x_face_info = list(pred_emo) + pred_feature
        x_face_info = np.array(x_face_info).reshape(1, -1)
        rf_probs = rf_model.predict_proba(x_face_info)[0]  # Ensure it's a 1D array

        # CNN Probability
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        cnn_probs = cnn_model.predict(img_array)[0]  # Also a 1D array

        # Combine CNN and Random Forest predictions with weighting
        final_pred = cnn_probs + rf_probs * [1.5, 5, 1, 2] + np.array([0.4, 0.1, 0, 0.42])
        predicted_class = np.argmax(final_pred)  # Get the index of the highest probability

        # Map the predicted class index to the corresponding label
        predicted_label = class_labels[predicted_class]

        # Convert the resized image to base64
        _, buffer = cv2.imencode('.jpg', resized_img)
        base64_resized_img = base64.b64encode(buffer).decode('utf-8')
        
        # Convert the resized image to base64
        _, l_buffer = cv2.imencode('.jpg', landmark_img)
        base64_landmark_img = base64.b64encode(l_buffer).decode('utf-8')

        # Return the prediction and the base64 image
        return jsonify({
            'prediction': predicted_label,
            'resized_img': base64_resized_img,
            'landmarked_img': base64_landmark_img
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=True)


