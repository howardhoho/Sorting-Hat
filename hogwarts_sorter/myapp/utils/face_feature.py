import dlib
import cv2
import numpy as np
import os





class FeatureExtractor:
    def __init__(self, img_cv, model_folder):
        self.img_cv = img_cv
        self.model_dir = os.path.join(os.path.dirname(__file__), model_folder)
        self.shape_predictor_path = os.path.join(self.model_dir, 'shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)

    def face_feature(self):
        # Convert the image to grayscale for landmark detection
        ff_img = self.img_cv.copy()
        gray = cv2.cvtColor(ff_img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.detector(gray)

        # Check if any face is detected
        if len(faces) == 0:
            return None

        # Process only the first detected face
        for face in faces:
            # Get facial landmarks
            landmarks = self.predictor(gray, face)

            # Align the face based on the eyes
            landmarks = self.align_face(ff_img, landmarks)

            # Recalculate facial dimensions after alignment using aligned_landmarks
            face_width = abs(landmarks[16].x - landmarks[0].x)
            face_height = abs(landmarks[27].y - landmarks[8].y)
            face_ratio = face_width / face_height

            # Normalize distances by face width
            normalized_distances = lambda x: x / face_width

            # --- Basic Distances ---
            eye_distance = normalized_distances(abs(landmarks[39].x - landmarks[42].x))
            nose_to_mouth_distance = normalized_distances(abs(landmarks[33].y - ((landmarks[62].y + landmarks[66].y) / 2)))
            nose_to_chin_distance = normalized_distances(abs(landmarks[33].y - landmarks[8].y))
            inner_eye_corner_distance = normalized_distances(abs(landmarks[39].x - landmarks[36].x))
            outer_eye_corner_distance = normalized_distances(abs(landmarks[42].x - landmarks[45].x))
            mouth_corner_distance = normalized_distances(abs(landmarks[48].x - landmarks[54].x))

            # --- Eyebrow-Eye Distance ---
            left_eyebrow_eye_distance = normalized_distances(abs(landmarks[19].y - landmarks[37].y))
            right_eyebrow_eye_distance = normalized_distances(abs(landmarks[24].y - landmarks[44].y))

            # --- Eyebrow Tilted Degree ---
            left_eyebrow_tilt = self.calculate_angle(landmarks[18], landmarks[19], landmarks[21])
            right_eyebrow_tilt = self.calculate_angle(landmarks[22], landmarks[24], landmarks[25])

            # --- Symmetry ---
            eye_symmetry = abs(normalized_distances(abs(landmarks[36].x - landmarks[39].x)) - normalized_distances(abs(landmarks[42].x - landmarks[45].x)))
            mouth_symmetry = abs(normalized_distances(abs(landmarks[48].y - landmarks[54].y)))
            eye_nose_symmetry = abs(normalized_distances(abs(landmarks[39].x - landmarks[33].x)) - normalized_distances(abs(landmarks[42].x - landmarks[33].x)))
            mouth_nose_symmetry = abs(normalized_distances(abs(landmarks[48].x - landmarks[33].x)) - normalized_distances(abs(landmarks[54].x - landmarks[33].x)))

            # --- Geometric Shapes ---
            triangle_area = self.calculate_triangle_area(landmarks[36], landmarks[45], landmarks[33])
            eye_mouth_quadrilateral_area = self.calculate_quadrilateral_area(landmarks[36], landmarks[45], landmarks[48], landmarks[54])

            # --- Curvature ---
            jawline_curvature = self.calculate_contour_length(landmarks, 0, 16)
            upper_lip_curvature = self.calculate_contour_length(landmarks, 48, 54)
            lower_lip_curvature = self.calculate_contour_length(landmarks, 55, 59)

            # --- Smile Intensity ---
            # Smile intensity is a combination of the mouth curvature and openness
            mouth_openness = normalized_distances(abs(landmarks[62].y - landmarks[66].y))
            smile_intensity = (upper_lip_curvature + lower_lip_curvature) * mouth_openness

            # --- Ratios ---
            eye_to_nose_ratio = eye_distance / normalized_distances(abs(landmarks[27].y - landmarks[33].y))
            eye_to_face_width_ratio = eye_distance / face_width
            mouth_to_face_width_ratio = mouth_corner_distance / face_width
            nose_to_face_height_ratio = nose_to_chin_distance / face_height

            # --- Angles ---
            nose_eye_angle = self.calculate_angle(landmarks[36], landmarks[33], landmarks[45])
            jawline_angle = self.calculate_angle(landmarks[4], landmarks[8], landmarks[12])

                 # --- Aspect Ratios ---
            mouth_width = normalized_distances(abs(landmarks[48].x - landmarks[54].x))
            mouth_height = normalized_distances(abs(landmarks[62].y - landmarks[66].y))
            if mouth_height != 0:
                mouth_aspect_ratio = mouth_width / mouth_height
            else:
                mouth_aspect_ratio = 0.2

            # --- Statistical Summary Features ---
            key_distances = [
                normalized_distances(abs(landmarks[36].x - landmarks[45].x)),  # Eye-to-eye
                normalized_distances(abs(landmarks[30].y - landmarks[8].y)),   # Nose to chin
                normalized_distances(abs(landmarks[33].y - landmarks[66].y))   # Nose to lower lip
            ]
            mean_distance = np.mean(key_distances)
            variance_distance = np.var(key_distances)

            return [
                face_ratio, eye_distance, nose_to_mouth_distance, nose_to_chin_distance, inner_eye_corner_distance,
                outer_eye_corner_distance, mouth_corner_distance, left_eyebrow_eye_distance, right_eyebrow_eye_distance,
                left_eyebrow_tilt, right_eyebrow_tilt, smile_intensity, eye_symmetry, mouth_symmetry, eye_nose_symmetry,
                mouth_nose_symmetry, triangle_area, eye_mouth_quadrilateral_area, jawline_curvature, upper_lip_curvature,
                lower_lip_curvature, eye_to_nose_ratio, eye_to_face_width_ratio, mouth_to_face_width_ratio,
                nose_to_face_height_ratio, nose_eye_angle, jawline_angle, mouth_aspect_ratio, mean_distance,
                variance_distance
            ]

        # Return None if no faces are detected
        print("No face detected!")
        return None


    def align_face(self, img_cv, landmarks):
        # Calculate the center points of the eyes using dlib's part() method
        left_eye_center = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye_center = np.array([landmarks.part(45).x, landmarks.part(45).y])

        # Calculate the angle between the two eye centers
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the center between the two eyes
        eye_center = (left_eye_center + right_eye_center) / 2

        # Get the rotation matrix for aligning the face
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1.0)

        # # Apply the affine transformation to align the face
        # img_aligned = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]))

        # Update the landmark positions after the transformation
        new_landmarks = []
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            new_landmarks.append(dlib.point(int(x_new), int(y_new)))

        # Return the updated landmark points as a list of dlib.point objects
        return new_landmarks


    def calculate_triangle_area(self, p1, p2, p3):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    def calculate_quadrilateral_area(self, p1, p2, p3, p4):
        area1 = self.calculate_triangle_area(p1, p2, p3)
        area2 = self.calculate_triangle_area(p1, p3, p4)
        return area1 + area2

    def calculate_contour_length(self, landmarks, start_idx, end_idx):
        length = 0
        for i in range(start_idx, end_idx):
            length += np.sqrt((landmarks[i].x - landmarks[i+1].x)**2 + 
                              (landmarks[i].y - landmarks[i+1].y)**2)
        return length

    def calculate_angle(self, p1, p2, p3):
        a = np.array([p1.x - p2.x, p1.y - p2.y])
        b = np.array([p3.x - p2.x, p3.y - p2.y])
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def face_landmark(self):
        lm_img = self.img_cv.copy()
        gray = cv2.cvtColor(lm_img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.detector(gray)

        # Process each detected face
        for face in faces:
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            marked_img = self.img_cv.copy()

            # Annotate the image with green circles at each landmark
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                marked_img = cv2.circle(marked_img, (x, y), 2, (0, 255, 0), -1)  # Green circles of radius 2
                
        marked_img_rgb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)


        return marked_img_rgb



