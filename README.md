Codebasics ML Course Health Insurance Prediction Project

![image alt](https://github.com/pierredeveloper/ML-Health-Insurance-Premium-Prediction/blob/main/README%20file%20banner%20Image..png?raw=true)

# 🏥 Health Insurance Premium Prediction

A machine learning project designed to predict health insurance premiums based on demographic, lifestyle, and medical history features. This project includes data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and deployment using Streamlit.

---

## 🚀 Project Overview
The goal of this project is to build a predictive model that estimates a person's health insurance premium. This can help insurance companies optimize pricing and individuals understand the factors influencing their premium.

The project includes:
- Data loading and preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model training (Linear Regression, Random Forest, XGBoost, etc.)
- Model evaluation
- Streamlit application for real-time prediction

---

## 📁 Project Structure
```
ML-Health-Insurance-Premium-Prediction/
│
├── data/                      # Raw dataset
├── notebooks/                 # Jupyter notebooks for EDA & training
├── artifacts/                 # Saved models & scalers
├── streamlit_app/             # Streamlit frontend
├── prediction_helper.py       # Helper functions for prediction
├── train_model.py             # Script to train the model
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🧠 ML Models Used
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor

Performance metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 🛠️ Installation
### 1. Clone the repository
```
git clone https://github.com/pierredeveloper/ML-Health-Insurance-Premium-Prediction.git
cd ML-Health-Insurance-Premium-Prediction
```

### 2. Create & activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App
```
streamlit run streamlit_app/app.py
```

The app will open automatically in your browser, allowing you to input data and get premium predictions instantly.

---

## 📊 Features Used for Prediction
- Age
- Sex
- BMI
- Smoking status
- Number of children
- Region
- Medical history

---

## 📦 Model Artifacts
Inside the `artifacts/` folder:
- `model_rest.joblib`
- `model_young.joblib`
- `scaler_rest.joblib`
- `scaler_young.joblib`

These are automatically loaded by `prediction_helper.py`.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss the proposed update.

---

## 📜 License
This project is licensed under the MIT License.

---

## 👤 Author
**Pierre Jean**  
Data Scientist & Developer  
GitHub: [pierredeveloper](https://github.com/pierredeveloper)

---

