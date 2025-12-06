import pandas as pd
from joblib import load
import os

# Correct path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "app", "artifacts")

# Load models
model_rest = load(os.path.join(ARTIFACT_PATH, "model_rest.joblib"))
model_young = load(os.path.join(ARTIFACT_PATH, "model_young.joblib"))

# Load scalers
scaler_rest = load(os.path.join(ARTIFACT_PATH, "scaler_rest.joblib"))
scaler_young = load(os.path.join(ARTIFACT_PATH, "scaler_young.joblib"))


# #Load models
# model_rest = load("app/artifacts/model_rest.joblib")
# model_young = load("app/artifacts/model_young.joblib")
#
# #Load scalers
# scaler_rest = load("app/artifacts/scaler_rest.joblib")
# scaler_young = load("app/artifacts/scaler_young.joblib")


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(d, 0) for d in diseases)

    max_score = 14
    return total_risk_score / max_score


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
        'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast',
        'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Fill dataframe
    df['age'] = input_dict['Age']
    df['number_of_dependants'] = input_dict['Number of Dependants']
    df['income_lakhs'] = input_dict['Income in Lakhs']
    df['genetical_risk'] = input_dict['Genetical Risk']
    df['insurance_plan'] = insurance_plan_encoding[input_dict['Insurance Plan']]
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    if input_dict['Gender'] == "Male":
        df['gender_Male'] = 1

    region_map = {
        "Northwest": "region_Northwest",
        "Southeast": "region_Southeast",
        "Southwest": "region_Southwest",
    }
    if input_dict['Region'] in region_map:
        df[region_map[input_dict['Region']]] = 1

    if input_dict['Marital Status'] == "Unmarried":
        df['marital_status_Unmarried'] = 1

    bmi_map = {
        "Obesity": "bmi_category_Obesity",
        "Overweight": "bmi_category_Overweight",
        "Underweight": "bmi_category_Underweight",
    }
    if input_dict['BMI Category'] in bmi_map:
        df[bmi_map[input_dict['BMI Category']]] = 1

    smoke_map = {
        "Occasional": "smoking_status_Occasional",
        "Regular": "smoking_status_Regular",
    }
    if input_dict['Smoking Status'] in smoke_map:
        df[smoke_map[input_dict['Smoking Status']]] = 1

    emp_map = {
        "Salaried": "employment_status_Salaried",
        "Self-Employed": "employment_status_Self-Employed",
    }
    if input_dict['Employment Status'] in emp_map:
        df[emp_map[input_dict['Employment Status']]] = 1

    df = handle_scaling(input_dict['Age'], df)

    return df


def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = 0  # dummy column
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis=1, inplace=True)

    return df


def predict(input_dict):
    df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(df)
    else:
        prediction = model_rest.predict(df)

    return int(prediction[0])







