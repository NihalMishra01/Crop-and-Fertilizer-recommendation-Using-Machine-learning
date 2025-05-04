# 🌾 Crop and Fertilizer Recommendation System using Machine Learning

This project provides an intelligent recommendation system that suggests the **most suitable crop** to grow based on environmental conditions and the **ideal fertilizer** to use based on crop nutrient deficiencies. Built using machine learning techniques and trained on real-world datasets, it is designed to assist farmers and agricultural stakeholders in making data-driven decisions.

---

## 📁 Project Structure

```
├── app.py                           # Streamlit/Flask application for user interface
├── crop_feature_scaler.pkl         # Scaler used for crop feature normalization
├── crop_features.pkl               # Selected features used in crop prediction
├── crop_label_encoder.pkl          # Encoder for crop labels
├── crop_recommendation.csv         # Dataset used for crop recommendation
├── crop_recommendation_model.pkl   # Trained model for crop prediction
├── fertilizer_feature_scaler.pkl   # Scaler used for fertilizer feature normalization
├── fertilizer_features.pkl         # Selected features used in fertilizer prediction
├── fertilizer_label_encoders.pkl   # Encoders for fertilizer classification
├── fertilizer_recommendation.csv   # Dataset used for fertilizer recommendation
├── fertilizer_recommendation_model.pkl # Trained model for fertilizer prediction
├── model_training.ipynb            # Notebook for model training and evaluation
├── preprocess.ipynb                # Notebook for data cleaning and preprocessing
├── train_model.py                  # Script to train and save ML models
└── README.md                       # Project documentation
```

---

## 🚀 How It Works

### Crop Recommendation
- **Input**: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall  
- **Output**: Most suitable crop to grow in given conditions  
- **Model**: Trained using Random Forest Classifier for high accuracy.

### Fertilizer Recommendation
- **Input**: Crop name, Nitrogen, Phosphorus, Potassium levels  
- **Output**: Ideal fertilizer suggestion  
- **Model**: Classification model trained to suggest fertilizers based on nutrient deficiency.

---

## 📊 Datasets Used

1. `crop_recommendation.csv`: Contains data on various environmental factors and suitable crops.
2. `fertilizer_recommendation.csv`: Contains fertilizer data mapped with crop nutrient requirements.

---

## 🔧 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/crop-fertilizer-recommendation.git
cd crop-fertilizer-recommendation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

*You may use [Streamlit](https://streamlit.io/) or [Flask](https://flask.palletsprojects.com/) depending on how `app.py` is set up.*

---

## 📈 Model Training

- Use `preprocess.ipynb` to clean and preprocess the data.
- Use `model_training.ipynb` or `train_model.py` to train models and generate `.pkl` files.
- Feature scaling and label encoding are stored separately for reproducibility.

---

## 🧠 Tech Stack

- **Language**: Python  
- **Libraries**: Scikit-learn, Pandas, NumPy, Streamlit/Flask  
- **Tools**: Jupyter Notebook, Pickle, Matplotlib (optional for visualizations)

---

## 📌 Features

- Dual recommendation: Crop and Fertilizer
- Pre-trained ML models for offline usage
- Scalable and modular code structure
- Clean UI for user input (in `app.py`)

---

## ✍️ Author

**Nihal Mishra**  
B.Tech CSE (AI & DS)  
[LinkedIn](https://www.linkedin.com/in/nihalmishraofficial)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
