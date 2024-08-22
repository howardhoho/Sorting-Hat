import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import dlib
import os

class EmoExtractor():
    
    def __init__(self, img_cv, model_folder):
        self.model_dir = os.path.join(os.path.dirname(__file__), model_folder)
        self.model_path = os.path.join(self.model_dir, 'affectnet_emo_model.h5')
        self.face_cascade_path = os.path.join(self.model_dir, 'haarcascade_frontalface_default.xml')
        self.model = tf.keras.models.load_model(self.model_path)
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)  # Load the face detector
        self.img_cv = img_cv
        

    def process_img(self, target_size=(96, 96), padding=20):
        emo_img = self.img_cv.copy()
        gray = cv2.cvtColor(emo_img, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)

        # Process each detected face
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            
            # Add padding around the detected face
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(emo_img.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(emo_img.shape[0] - y_padded, h + 2 * padding)
            
            # Crop the face with padding from the original color image
            face_cropped = emo_img[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
            
            h_cropped, w_cropped = face_cropped.shape[:2]
            scale = min(target_size[0] / h_cropped, target_size[1] / w_cropped)
            
            # Resize while maintaining aspect ratio
            resized_face = cv2.resize(face_cropped, (int(w_cropped * scale), int(h_cropped * scale)))

            # Create a black (or white) canvas of the target size (96x96)
            canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)  # Black canvas
            y_offset = (target_size[0] - resized_face.shape[0]) // 2
            x_offset = (target_size[1] - resized_face.shape[1]) // 2
            
            # Place the resized face in the center of the 96x96 canvas
            canvas[y_offset:y_offset+resized_face.shape[0], x_offset:x_offset+resized_face.shape[1]] = resized_face
            
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

            # Return the processed face (only one face will be returned)
            return canvas_rgb

    
    def predict(self, resized_image):
        img_array = image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = self.model.predict(img_array)
        return prediction[0]

            
        
        
        