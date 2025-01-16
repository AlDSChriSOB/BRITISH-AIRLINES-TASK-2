#!/usr/bin/env python
# coding: utf-8

# # Task 2
# 
# ---
# 
# ## Predictive modeling of customer bookings
# 
# This Jupyter notebook includes some code to get you started with this predictive modeling task. We will use various packages for data manipulation, feature engineering and machine learning.
# 
# ### Exploratory data analysis
# 
# First, we must explore the data in order to better understand what we have and the statistical properties of the dataset.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import shap


# In[2]:


df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
df.head()


# In[3]:


df


# The `.head()` method allows us to view the first 5 rows in the dataset, this is useful for visual inspection of our columns

# In[4]:


df.info()


# The `.info()` method gives us a data description, telling us the names of the columns, their data types and how many null values we have. Fortunately, we have no null values. It looks like some of these columns should be converted into different data types, e.g. flight_day.
# 
# To provide more context, below is a more detailed data description, explaining exactly what each column means:
# 
# - `num_passengers` = number of passengers travelling
# - `sales_channel` = sales channel booking was made on
# - `trip_type` = trip Type (Round Trip, One Way, Circle Trip)
# - `purchase_lead` = number of days between travel date and booking date
# - `length_of_stay` = number of days spent at destination
# - `flight_hour` = hour of flight departure
# - `flight_day` = day of week of flight departure
# - `route` = origin -> destination flight route
# - `booking_origin` = country from where booking was made
# - `wants_extra_baggage` = if the customer wanted extra baggage in the booking
# - `wants_preferred_seat` = if the customer wanted a preferred seat in the booking
# - `wants_in_flight_meals` = if the customer wanted in-flight meals in the booking
# - `flight_duration` = total duration of flight (in hours)
# - `booking_complete` = flag indicating if the customer completed the booking
# 
# Before we compute any statistics on the data, lets do any necessary data conversion

# In[5]:


df["flight_day"].unique()


# In[6]:


mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df["flight_day"] = df["flight_day"].map(mapping)


# In[7]:


df["flight_day"].unique()


# In[8]:


df.describe()


# The `.describe()` method gives us a summary of descriptive statistics over the entire dataset (only works for numeric columns). This gives us a quick overview of a few things such as the mean, min, max and overall distribution of each column.
# 
# From this point, you should continue exploring the dataset with some visualisations and other metrics that you think may be useful. Then, you should prepare your dataset for predictive modelling. Finally, you should train your machine learning model, evaluate it with performance metrics and output visualisations for the contributing variables. All of this analysis should be summarised in your single slide.

# In[9]:


print("\nMissing Values:")


# In[10]:


print(df.isnull().sum())


# No missing value 

# In[11]:


# Data Exploration
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution:")
print(df['booking_complete'].value_counts(normalize=True))


# In[12]:


from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
df_encoded = df.copy()
for col in df.select_dtypes(include=['object']).columns:  # Select categorical columns
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Correlation Heatmap with Encoded Data
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[13]:


# Visualizing Class Distribution
sns.countplot(x='booking_complete', data=df)
plt.title('Target Variable Distribution')
plt.show()


# In[14]:


# Convert Categorical Features to Numeric
categorical_features = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']
le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])


# In[15]:


# Feature Engineering
df['weekend_flight'] = df['flight_day'].apply(lambda x: 1 if x in ['Sat', 'Sun'] else 0)
df['long_stay'] = df['length_of_stay'].apply(lambda x: 1 if x > 14 else 0)


# In[16]:


# Distribution of Flight Hours
plt.figure(figsize=(10,5))
sns.histplot(df['flight_hour'], bins=24, kde=True)
plt.title('Distribution of Flight Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()


# In[17]:


# Boxplot for Length of Stay
plt.figure(figsize=(10,5))
sns.boxplot(x='booking_complete', y='length_of_stay', data=df)
plt.title('Length of Stay vs Booking Completion')
plt.show()


# In[18]:


# Countplot for Sales Channel
plt.figure(figsize=(8,5))
sns.countplot(x='sales_channel', hue='booking_complete', data=df)
plt.title('Booking Completion by Sales Channel')
plt.show()


# In[19]:


# Histogram of Length of Stay before and after transformation
plt.figure(figsize=(10,5))
sns.histplot(df['length_of_stay'], bins=30, kde=True, color='blue', label='Original')
sns.histplot(df['long_stay'], bins=3, kde=True, color='red', label='Transformed')
plt.legend()
plt.title('Effect of Long Stay Feature Engineering')
plt.show()


# In[20]:


# Countplot for Weekend Flights
plt.figure(figsize=(8,5))
sns.countplot(x='weekend_flight', hue='booking_complete', data=df)
plt.title('Impact of Weekend Flights on Booking Completion')
plt.show()


# BUILDING MODEL FOR PREDICTION OF TARGET OUTCOME

# In[21]:


# Splitting Data
X = df.drop(columns=['booking_complete'])
y = df['booking_complete']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Scaling Numeric Features
scaler = StandardScaler()
numeric_features = ['purchase_lead', 'length_of_stay', 'flight_hour', 'flight_duration']
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])


# In[23]:


# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[24]:


# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[25]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Initialize Stratified KFold (preserves class distribution in each fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store scores from cross-validation
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


# Cross-validation loop
for train_idx, val_idx in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Fit the model
    rf.fit(X_train_fold, y_train_fold)
    
    # Predict on validation fold
    y_pred_fold = rf.predict(X_val_fold)
    
    # Calculate metrics for this fold
    accuracy_scores.append(accuracy_score(y_val_fold, y_pred_fold))
    precision_scores.append(precision_score(y_val_fold, y_pred_fold))
    recall_scores.append(recall_score(y_val_fold, y_pred_fold))
    f1_scores.append(f1_score(y_val_fold, y_pred_fold))

# Average scores from cross-validation
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

# Output cross-validation results
print(f"Cross-Validation Results (Average scores over {cv.get_n_splits()} folds):")
print(f"Accuracy: {avg_accuracy:.4f}")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1-Score: {avg_f1:.4f}")

# Train final model on the whole training set
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


# In[26]:


# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[27]:


# Classification Report
print("\nClassification Report on Test Data:")
print(classification_report(y_test, y_pred))


# In[28]:


#Save the Model Using pickle

import pickle

# Save the trained model
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model saved successfully!")


# In[29]:


# Feature Importance
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()


# In[30]:


# Save Results
feature_importances.to_csv("feature_importance.csv", index=False)

print("Model training and evaluation completed successfully!")


# In[31]:


# Load the trained model for Inference
with open("random_forest_model.pkl", "rb") as f:
    loaded_rf = pickle.load(f)

# Make predictions
y_pred = loaded_rf.predict(X_test)

