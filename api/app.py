from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open('le.pkl', 'rb') as f:
    le = pickle.load(f)

class CustomerData(BaseModel):
    tenure: int
    monthlycharges: float
    totalcharges: float
    total_services: int
    avg_revenue_per_month: float
    contract: str
    paymentmethod: str
    gender: str
    partner: str
    dependents: str
    seniorcitizen: int
    tenure_segment: str

@app.get('/')
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post('/predict')
def predict(data: CustomerData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Label encode tenure_segment
    input_df['tenure_segment'] = le.transform(input_df['tenure_segment'])
    
    # Define column groups
    nominal_cols = ['gender', 'partner', 'dependents', 
                    'contract', 'paymentmethod']
    num_cols = ['tenure', 'monthlycharges', 'totalcharges', 
                'total_services', 'avg_revenue_per_month', 'tenure_segment']
    
    # OneHotEncode nominal columns
    encoded = ohe.transform(input_df[nominal_cols])
    encoded_df = pd.DataFrame(
        encoded, 
        columns=ohe.get_feature_names_out(nominal_cols)
    )
    
    # Drop original nominal cols and add encoded ones
    input_df = pd.concat([
        input_df.drop(columns=nominal_cols).reset_index(drop=True), 
        encoded_df.reset_index(drop=True)
    ], axis=1)
    
    # Scale numerical columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Reorder columns to match training order
    correct_order = ['seniorcitizen', 'tenure', 'tenure_segment', 
                     'total_services', 'avg_revenue_per_month', 
                     'monthlycharges', 'totalcharges',
                     'gender_Female', 'gender_Male', 
                     'partner_No', 'partner_Yes', 
                     'dependents_No', 'dependents_Yes',
                     'contract_Month-to-month', 'contract_One year', 'contract_Two year',
                     'paymentmethod_Bank transfer', 'paymentmethod_Credit card', 
                     'paymentmethod_Electronic check', 'paymentmethod_Mailed check']
    
    input_df = input_df[correct_order]
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    
    # Risk tier
    if prob > 0.7:
        risk = 'High'
    elif prob > 0.4:
        risk = 'Medium'
    else:
        risk = 'Low'
    
    return {
        'churn_probability': round(float(prob), 4),
        'churn_prediction': int(prediction),
        'risk_tier': risk,
        'message': 'This customer is likely to churn' if prediction == 1 else 'This customer is unlikely to churn'
    }
