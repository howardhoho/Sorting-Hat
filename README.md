# Sorting-Hat -- CNN-Based  Face Classification Web App
Sort the your uploaded face images to one of the four houses in Hogwarts: "Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin" using CNN based models and face features extraction methods.


# Project Information
The objective is to sort one into one of four Hogwarts houses using 2 CNN models and the combination of Random Forest for face feature extractions.
The app is hosted on AWS EC2 with RDS and MySQL for data storage.

# Website
➡️ [Click here to the website!](http://3.18.180.241:5000/)<br />
<br />
Users upload the face image, and they can get the prediction of the house and the landmarked uploaded image. 
![website screenshot](https://github.com/user-attachments/assets/ac6cd198-ac38-4da2-9d17-42f7b2f5543d)


# Main Models

- Neural Networks<br />
  1. <ins>Emotion Recognition CNN Model</ins><br />
    Convolutional Layers: 4<br />
   It was trained using the Affectnet dataset: https://www.kaggle.com/datasets/noamsegal/affectnet-training-data.
   It was built with TensorFlow and Keras for classifying facial emotions from images. The model was trained on the AffectNet dataset, which included images categorized into 8 emotional states such as 'angry', 'sad', etc

  2. <ins>Harry Potter Characters Face Images CNN Model</ins><br />
    Convolutional Layers: 4<br />
    It was trained using approximately 2000 images of the 253 Harry Potter actors. Data Augmentation was adoped for dataset imbalance due to the lack of "HufflePuff" and "Ravenclaw", as well as to increase the training data.

- Other Models<br />
  1. <ins>Face Features Extraction </ins><br />
    This is a method to extract the face features of the Harry Potter characters using the landmarks from Dlib. The landmarks' coordinates were grouped by different face features for calculations. For example the "eye_distance", the "jawline_curvature" or the "upper_lip_curvature" etc. Those extracted features were normalized by the length and angle for further use.
  

  2. <ins>Combined Random Forest Model</ins><br />
     The extracted face features and the result of the proporton analysis from the Emotion Recognitoon Model were served as the input for the Random Forest Model for Classsfication. There were 38 features in total, including 8 from emotion proportion, and 30 from face features. The prediction is generated as a list of ratio of the four houses as the output.
- Final Prediciton<br />
  The final prediciton combined the results from the Random Forest(Face Features + Emotion Recognition CNN) and the Character Images CNN. Hyperparameter tuning and regularization were performed for the final prediction.





