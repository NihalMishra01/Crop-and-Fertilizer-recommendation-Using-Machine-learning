import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Create directory for models
os.makedirs("models", exist_ok=True)

# Load datasets
crop_df = pd.read_csv("crop_recommendation.csv")
fertilizer_df = pd.read_csv("fertilizer_recommendation.csv")

# ----------- Preprocess Fertilizer Dataset -----------
fertilizer_df_clean = fertilizer_df.copy()
label_encoders = {}

# Encode categorical columns: Soil Type, Crop Type, Fertilizer Name
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    le = LabelEncoder()
    fertilizer_df_clean[col] = le.fit_transform(fertilizer_df_clean[col])
    label_encoders[col] = le

# Create feature scaler for fertilizer data
fertilizer_features = ['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
fertilizer_scaler = StandardScaler()
fertilizer_df_clean[fertilizer_features] = fertilizer_scaler.fit_transform(fertilizer_df_clean[fertilizer_features])

# ----------- Preprocess Crop Dataset -----------
crop_df_clean = crop_df.copy()
crop_label_encoder = LabelEncoder()
crop_df_clean['label'] = crop_label_encoder.fit_transform(crop_df_clean['label'])

# Create feature scaler for crop data
crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
crop_scaler = StandardScaler()
crop_df_clean[crop_features] = crop_scaler.fit_transform(crop_df_clean[crop_features])

# ----------- Train Crop Recommendation Model -----------
X_crop = crop_df_clean[crop_features]
y_crop = crop_df_clean['label']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(random_state=42)
crop_model.fit(Xc_train, yc_train)

# ----------- Train Fertilizer Recommendation Model -----------
X_fert = fertilizer_df_clean[fertilizer_features]
y_fert = fertilizer_df_clean['Fertilizer Name']

Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42)
fertilizer_model = RandomForestClassifier(random_state=42)
fertilizer_model.fit(Xf_train, yf_train)

# ----------- Save Models and Preprocessing Objects -----------
# Save crop-related objects
joblib.dump(crop_model, "crop_recommendation_model.pkl")
joblib.dump(crop_scaler, "crop_feature_scaler.pkl")
joblib.dump(crop_label_encoder, "crop_label_encoder.pkl")
joblib.dump(crop_features, "crop_features.pkl")

# Save fertilizer-related objects
joblib.dump(fertilizer_model, "fertilizer_recommendation_model.pkl")
joblib.dump(fertilizer_scaler, "fertilizer_feature_scaler.pkl")
joblib.dump(label_encoders, "fertilizer_label_encoders.pkl")
joblib.dump(fertilizer_features, "fertilizer_features.pkl")

# Print model performance
print("\nModel Performance:")
print("Crop Model Accuracy:", crop_model.score(Xc_test, yc_test))
print("Fertilizer Model Accuracy:", fertilizer_model.score(Xf_test, yf_test))

print("\n✅ All models and preprocessing objects saved successfully!")
print("Saved files:")
print("- crop_recommendation_model.pkl")
print("- crop_feature_scaler.pkl")
print("- crop_label_encoder.pkl")
print("- crop_features.pkl")
print("- fertilizer_recommendation_model.pkl")
print("- fertilizer_feature_scaler.pkl")
print("- fertilizer_label_encoders.pkl")
print("- fertilizer_features.pkl")
