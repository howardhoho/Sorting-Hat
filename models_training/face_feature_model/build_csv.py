import csv
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import dlib
import os
from face_emo import EmoExtractor
from face_feature import FeatureExtractor


# Define the header
header = ["anger", "contempt", "disgust", "fear", "happy","neutral", "sad", "surprise",
          "face_ratio", "eye_distance", "nose_to_mouth_distance", "nose_to_chin_distance", 
          "inner_eye_corner_distance", "outer_eye_corner_distance", "mouth_corner_distance", 
          "left_eyebrow_eye_distance", "right_eyebrow_eye_distance", "left_eyebrow_tilt", 
          "right_eyebrow_tilt", "smile_intensity", "eye_symmetry", "mouth_symmetry", 
          "eye_nose_symmetry", "mouth_nose_symmetry", "triangle_area", "eye_mouth_quadrilateral_area", 
          "jawline_curvature", "upper_lip_curvature", "lower_lip_curvature", "eye_to_nose_ratio", 
          "eye_to_face_width_ratio", "mouth_to_face_width_ratio", "nose_to_face_height_ratio", 
          "nose_eye_angle", "jawline_angle", "mouth_aspect_ratio", "mean_distance", "variance_distance", "house"
          ]

# CSV file path
csv_file = 'aug-house-train-model.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header first
    writer.writerow(header)
    
    # Define the path to the main folder containing the four subfolders
    main_folder = 'aug_images'

    # List all the subfolders inside the main folder
    subfolders = [os.path.join(main_folder, subfolder) for subfolder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, subfolder))]

    # Loop through each subfolder
    for subfolder in subfolders:
        for file_name in os.listdir(subfolder):
            img_path = os.path.join(subfolder, file_name)
                
            emo_extractor = EmoExtractor(img_path)
            feature_extractor = FeatureExtractor(img_path)
            
            processed_img = emo_extractor.process_img()
            
            if processed_img is not None:
                prediction = emo_extractor.predict(processed_img)
                features = feature_extractor.face_feature()
                house_name = [os.path.basename(subfolder)]
                face_info_list = list(prediction)+ list(features)+house_name
                writer.writerow(face_info_list)
                
        
