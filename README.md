# BRITISH-AIRLINES-TASK-2
Customer Booking Prediction
Overview
This project involves training a machine learning model to predict whether a customer will complete a flight booking based on various attributes. The dataset is preprocessed, analyzed, and used to train a Random Forest Classifier.

Requirements
To run this project, ensure you have the following libraries installed:

pip install pandas numpy matplotlib seaborn scikit-learn shap
Dataset
The dataset includes features like:

Number of passengers

Booking origin

Flight duration

Extra baggage, preferred seat, in-flight meals

Flight day and hour

Target variable: booking_complete

Steps to Run
Load the dataset and perform Exploratory Data Analysis (EDA).

Encode categorical variables and create new features.

Train the Random Forest Classifier.

Evaluate the model using accuracy, precision, recall, and F1-score.

Save the model and use it for inference.

Visualize feature importance.

Model Training
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
Save and Load Model
import pickle
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)
To load the model:

with open("random_forest_model.pkl", "rb") as f:
    loaded_rf = pickle.load(f)
Skills Demonstrated
Data Analysis & Preprocessing:
Handled missing values and categorical encoding (LabelEncoder).

Performed exploratory data analysis (pandas, seaborn, matplotlib).

Engineered new features to improve model performance.

Machine Learning & Model Evaluation:
Implemented Random Forest Classifier.

Used cross-validation for model reliability.

Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrix.

Applied feature importance analysis for interpretability (shap).

Data Visualization:
Created heatmaps, histograms, count plots, and boxplots to understand data trends.

Visualized feature importance to identify key factors affecting bookings.

Model Deployment & Saving:
Saved and loaded the trained model using pickle.

Ensured the model is reusable for future inference tasks.

Programming & Technical Skills:
Python (pandas, numpy, seaborn, matplotlib, scikit-learn, shap)

Machine Learning (RandomForestClassifier, cross_val_score, GridSearchCV)

Data Processing (LabelEncoder, StandardScaler)

Model Deployment (pickle for model persistence)

This project showcases expertise in data preprocessing, machine learning, model evaluation, and visualization, making it a valuable addition to an AI and data science portfolio.
