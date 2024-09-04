import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import dlib

class EmoExtractor():
    
    def __init__(self, img_path):
        self.model = tf.keras.models.load_model('model/affectnet_emo_model.h5')
        self.face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')  
        self.img_path = img_path
        

    def process_img(self, target_size=(96, 96), padding=20):
        # Load the image
        img = cv2.imread(self.img_path)
        
        # Check if the image was successfully loaded
        if img is None:
            print(f"Error loading image: {self.img_path}")
            return None


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)

        # Process each detected face
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            
            # Add padding around the detected face
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(img.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(img.shape[0] - y_padded, h + 2 * padding)
            
            # Crop the face with padding from the original color image
            face_cropped = img[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]

            # Resize the face to maintain aspect ratio and fit within 96x96
            # First, get the scaling factor to fit within the target size
            h_cropped, w_cropped = face_cropped.shape[:2]
            scale = min(target_size[0] / h_cropped, target_size[1] / w_cropped)

            # Resize while maintaining aspect ratio
            resized_face = cv2.resize(face_cropped, (int(w_cropped * scale), int(h_cropped * scale)))

            # Create a black (or white) canvas of the target size (96x96)
            canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)  # Black canvas

            # Calculate the placement to center the resized face in the 96x96 canvas
            y_offset = (target_size[0] - resized_face.shape[0]) // 2
            x_offset = (target_size[1] - resized_face.shape[1]) // 2

            # Place the resized face in the center of the 96x96 canvas
            canvas[y_offset:y_offset+resized_face.shape[0], x_offset:x_offset+resized_face.shape[1]] = resized_face

            # Return the processed face (only one face will be returned)
            return canvas

    
    def predict(self, resized_image):
        img_array = image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = self.model.predict(img_array)
        return prediction[0]

            
    def merge_info(self, list1, list2):
        face_info_list = list1+list2
        return face_info_list
        
        
        