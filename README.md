# 🌾 **Crop and Fertilizer Recommendation System** using Machine Learning



Welcome to the **Crop and Fertilizer Recommendation System**, designed to help farmers and agricultural stakeholders make data-driven decisions for growing crops and choosing the right fertilizers.

This system uses machine learning models trained on real-world agricultural datasets to predict the best crop and fertilizer based on environmental conditions and nutrient requirements.

---

## 📊 **Project Overview**

| **Feature**                          | **Description**                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------|
| **Model Type**                       | Machine Learning (Random Forest Classifier for Crop Prediction and Classification for Fertilizers)    |
| **Primary Focus**                    | Predict the most suitable crop to grow and the ideal fertilizer to use based on environmental factors |
| **Tech Stack**                       | Python, Streamlit, Scikit-learn, Pandas, NumPy                                                       |
| **Dependencies**                      | Install via `requirements.txt`                                                                      |
| **Input Data**                       | Crop and Fertilizer data from CSVs, Preprocessed in Jupyter Notebooks                                |
| **Output**                           | Recommended crop and fertilizer based on the user's inputs                                           |

---

## 📁 **Project Structure**

```
├── app.py                           # Streamlit application for UI
├── crop_feature_scaler.pkl         # Scaler for crop feature normalization
├── crop_features.pkl               # Features for crop prediction model
├── crop_label_encoder.pkl          # Label encoder for crops
├── crop_recommendation.csv         # Dataset for crop recommendations
├── crop_recommendation_model.pkl   # Trained model for crop recommendation
├── fertilizer_feature_scaler.pkl   # Scaler for fertilizer feature normalization
├── fertilizer_features.pkl         # Features for fertilizer prediction model
├── fertilizer_label_encoders.pkl   # Label encoder for fertilizers
├── fertilizer_recommendation.csv   # Dataset for fertilizer recommendations
├── fertilizer_recommendation_model.pkl # Trained model for fertilizer recommendation
├── model_training.ipynb            # Model training and evaluation notebook
├── preprocess.ipynb                # Data preprocessing notebook
├── train_model.py                  # Script for training the models
└── README.md                       # Project documentation
```

---

## 🚀 **How to Run the Application**

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/crop-fertilizer-recommendation.git
cd crop-fertilizer-recommendation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

This command will start a local server and open the app in your browser where you can interact with the recommendation system.

---

## 📈 **Model Training and Data Preprocessing**

### Notebooks:
- **`preprocess.ipynb`**: Used for cleaning and preprocessing the input data.
- **`model_training.ipynb`**: Trains the crop and fertilizer recommendation models and stores them as `.pkl` files.

---

## 🧠 **Tech Stack**

| **Category**    | **Technology**           |
|-----------------|--------------------------|
| **Programming Language** | Python                |
| **Web Framework**        | Streamlit              |
| **Machine Learning**     | Scikit-learn, Pandas   |
| **Data Preprocessing**   | NumPy, Pandas         |
| **Model Serialization**  | Pickle                |

---

## 🎨 **Features**

- **Dual Recommendation**: Predicts both suitable crops and fertilizers based on environmental data.
- **Pre-trained Models**: Use pre-trained models for crop and fertilizer prediction.
- **Modular Design**: Scalable and easily extendable for more crops or fertilizers.
- **Interactive UI**: Streamlit-based interface for easy data input and recommendations.

---

## ✍️ **Author**

**Nihal Mishra**  
B.Tech CSE (AI & DS)  
[LinkedIn](https://www.linkedin.com/in/nihalmishraofficial)

---

## 📜 **License**

This project is open-source and available under the [MIT License](LICENSE).

---

## 🔖 **Badges**

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

