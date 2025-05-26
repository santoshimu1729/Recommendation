from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import pairwise_distances # Added scikit-learn import based on likely usage from requirements

app = FastAPI()

# Configure CORS
origins = [
    "*"  # Allows all origins - VERY permissive and NOT recommended for production
    # "http://localhost:3000", # Example: Allow requests from your frontend running on port 3000
    # "https://your-frontend-domain.com", # Example: Allow requests from your production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input data structure - Ensure field names match the new sample request
class InputData(BaseModel):
    Age: int
    Gender: str
    Marital_Status: str
    Profession: str
    Driving_Experience: int
    Past_Claim_History: int
    Past_Traffic_Violations: int
    Vehicle_Age: int
    Vehicle_Type: str
    Vehicle_Make: str
    Vehicle_Model: str
    Vehicle_Value: int
    Primary_Use: str
    Estimated_Annual_Mileage: int
    Safety_Features_ABS: str
    Safety_Features_Airbags: str
    Safety_Features_ESC: str
    Parking_Location: str
    Geographic_Location: str
    # 'Owns Boat' is missing in the new request, keeping it optional if needed
    Owns_Boat: str = None

# Define the response structure
class PredictionItem(BaseModel):
    coverage: str
    predicted_value: int
    factors: Dict[str, float]

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]

# Load the trained models and feature columns
try:
    model_package = joblib.load("finalLearning.pkl")
    loaded_models = model_package['models']
    loaded_feature_columns = model_package['feature_columns']
    loaded_categorical_features = model_package['categorical_features']
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Error: finalLearning.pkl not found.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model package: {e}")

# Ensure the CATEGORICAL_FEATURES list matches the training script's original categorical columns
CATEGORICAL_FEATURES_FASTAPI = [
    'Gender', 'Marital_Status', 'Profession',
    'Vehicle_Type', 'Vehicle_Make', 'Vehicle_Model',
    'Primary_Use', 'Safety_Features_ABS', 'Safety_Features_Airbags',
    'Safety_Features_ESC', 'Blind Spot Monitoring', 'Lane Keeping Assist', 'Automatic Emergency Braking',
    'Parking_Location', 'Geographic_Location', 'Owns Boat' # Keeping 'Owns Boat' for consistency
]

# Mapping for the keys in the new request format to the keys expected by the trained model
INPUT_FIELD_MAPPING = {
    "Age": "Age",
    "Gender": "Gender",
    "Marital_Status": "Marital Status",
    "Profession": "Profession/Occupation",
    "Driving_Experience": "Driving Experience (in years)",
    "Past_Claim_History": "Past Claim History",
    "Past_Traffic_Violations": "Past Traffic Violations",
    "Vehicle_Age": "Vehicle Age (in years)",
    "Vehicle_Type": "Vehicle Type/Category",
    "Vehicle_Make": "Vehicle Make",
    "Vehicle_Model": "Vehicle Model",
    "Vehicle_Value": "Vehicle Value/Price",
    "Primary_Use": "Primary Use of Vehicle",
    "Estimated_Annual_Mileage": "Estimated Annual Mileage",
    "Safety_Features_ABS": "ABS",
    "Safety_Features_Airbags": "Airbags",
    "Safety_Features_ESC": "Electronic Stability Control",
    "Parking_Location": "Parking Location",
    "Geographic_Location": "Geographic Location/City"
    # 'Owns Boat' will be handled if present in the request
}

@app.get("/")
async def welcome():
    return {"message": "Welcome to recommendation system"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(data: InputData):
    try:
        input_dict = data.dict(exclude_none=True)
        # Create a DataFrame with columns matching the training data
        mapped_input_data = {}
        for request_key, request_value in input_dict.items():
            if request_key in INPUT_FIELD_MAPPING:
                mapped_input_data[INPUT_FIELD_MAPPING[request_key]] = [request_value]
            elif request_key == 'Owns_Boat':
                mapped_input_data['Owns Boat'] = [request_value] # Handle 'Owns Boat' if provided

        input_df = pd.DataFrame(mapped_input_data)

        # Preprocess the input data
        input_encoded = pd.get_dummies(input_df, columns=[INPUT_FIELD_MAPPING.get(col) for col in CATEGORICAL_FEATURES_FASTAPI if INPUT_FIELD_MAPPING.get(col) in input_df.columns or col == 'Owns Boat' and 'Owns Boat' in input_df.columns], drop_first=True)

        # Reindex to align with the training data's columns
        final_input = input_encoded.reindex(columns=loaded_feature_columns, fill_value=0)

        predictions = []

        for target_col, model in loaded_models.items():
            prediction = model.predict(final_input)[0]
            predicted_risk_score = int(np.clip(np.round(prediction), 1, 5))

            # Get global feature importance
            importance = model.feature_importances_
            feature_importance_dict_full = {loaded_feature_columns[i]: importance[i] for i in range(len(loaded_feature_columns))}

            # Filter contributing factors to only include importance of input features
            input_features_for_factors = {}
            for request_key in input_dict.keys():
                original_feature_name = INPUT_FIELD_MAPPING.get(request_key)
                if original_feature_name:
                    for encoded_feature_name, importance_score in feature_importance_dict_full.items():
                        if original_feature_name.replace('_', ' ').lower() in encoded_feature_name.lower():
                            input_features_for_factors[request_key] = importance_score
                            break
                elif request_key == 'Owns_Boat':
                    for encoded_feature_name, importance_score in feature_importance_dict_full.items():
                        if 'owns boat' in encoded_feature_name.lower():
                            input_features_for_factors[request_key] = importance_score
                            break

            # Create the prediction item
            coverage_name = target_col.replace("Risk Score - ", "")
            prediction_item = PredictionItem(
                coverage=coverage_name,
                predicted_value=predicted_risk_score,
                factors=input_features_for_factors
            )
            predictions.append(prediction_item)

        return PredictionResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sample Request (to test with a client like Postman or curl)
"""
{
    "Age": 30,
    "Gender": "Male",
    "Marital_Status": "Married",
    "Profession": "Engineer",
    "Driving_Experience": 10,
    "Past_Claim_History": 0,
    "Past_Traffic_Violations": 0,
    "Vehicle_Age": 2,
    "Vehicle_Type": "Sedan",
    "Vehicle_Make": "Toyota",
    "Vehicle_Model": "Camry",
    "Vehicle_Value": 35000,
    "Primary_Use": "Commuting",
    "Estimated_Annual_Mileage": 12000,
    "Safety_Features_ABS": "Yes",
    "Safety_Features_Airbags": "Yes",
    "Safety_Features_ESC": "Yes",
    "Parking_Location": "Secured Garage",
    "Geographic_Location": "Seattle, WA"
}
"""

# To run this FastAPI application:
# 1. Save the code as a Python file (e.g., main.py).
# 2. Make sure you have FastAPI and Uvicorn installed: pip install fastapi uvicorn
# 3. Run the application from your terminal: uvicorn main:app --reload
# 4. You can then send POST requests to http://127.0.0.1:8000/predict with the sample request data.