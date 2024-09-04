import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib


data = pd.read_csv('aug-house-train-model.csv')

# Separate features and target
X = data.drop(columns=['house'])
y = data['house']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to the training set to oversample minority classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define Random Forest Classifier
predict_random_forest = RandomForestClassifier()

print("running......")

# Hyperparameter grid
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 6],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', {0: 1, 1: 1, 2: 1, 3: 1}]  
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(predict_random_forest, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Cross-validation accuracy on the balanced dataset
cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
print(f"Best Random Forest - Average CV Accuracy: {np.mean(cv_scores):.4f}")

# Test set evaluation
y_pred = best_model.predict(X_test)
print(f"Best Random Forest - Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the trained model and label encoder for future use
joblib.dump(best_model, 'aug_random_forest_house_classifier.pkl')
joblib.dump(label_encoder, 'aug_label_encoder.pkl')

