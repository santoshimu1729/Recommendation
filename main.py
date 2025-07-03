from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from sklearn.metrics import pairwise_distances
import datetime

app = FastAPI()

# Configure CORS
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input data structure for original /predict endpoint
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
    Owns_Boat: str = None

# Define the response structure for original /predict endpoint
class PredictionItem(BaseModel):
    coverage: str
    predicted_value: int
    factors: Dict[str, float]

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]

# --- NEW MODELS FOR /predict_coverage ENDPOINT ---

# Define the input data structure for the new /predict_coverage endpoint
class CoverageRequest(BaseModel):
    Policy_ID: str
    NewVehicle: str
    VehicleType: str = Field(..., description="Possible values: 4wheeler or 2wheeler")
    VehicleMake: str
    VehicleModel: str
    VehicleYear: int
    VehicleIDV: int = Field(..., description="Insured Declared Value")
    VehicleFuelType: str = Field(..., description="Possible values: Petrol, Diesel, CNG, Hybrid")
    City: str
    PostalCode: int

# Define the recommendation item structure for the new /predict_coverage endpoint
class CoverageRecommendationItem(BaseModel):
    coveragesPatternCode: List[str] = Field(..., description="List of recommended coverages")
    reason: str

# Define the response structure for the new /predict_coverage endpoint
class CoverageRecommendationResponse(BaseModel):
    recommendations: List[CoverageRecommendationItem]

# --- END NEW MODELS ---

# Load the trained models and feature columns
try:
    model_package = joblib.load("finalLearning.pkl")
    loaded_models = model_package['models']
    loaded_feature_columns = model_package['feature_columns']
    loaded_categorical_features = model_package['categorical_features']
except FileNotFoundError:
    print("Warning: finalLearning.pkl not found. /predict endpoint will not function correctly.")
    loaded_models = {}
    loaded_feature_columns = []
    loaded_categorical_features = []
except Exception as e:
    print(f"Warning: Error loading model package: {e}. /predict endpoint will not function correctly.")
    loaded_models = {}
    loaded_feature_columns = []
    loaded_categorical_features = []


# Ensure the CATEGORICAL_FEATURES list matches the training script's original categorical columns
CATEGORICAL_FEATURES_FASTAPI = [
    'Gender', 'Marital_Status', 'Profession',
    'Vehicle_Type', 'Vehicle_Make', 'Vehicle_Model',
    'Primary_Use', 'Safety_Features_ABS', 'Safety_Features_Airbags',
    'Safety_Features_ESC', 'Blind Spot Monitoring', 'Lane Keeping Assist', 'Automatic Emergency Braking',
    'Parking_Location', 'Geographic_Location', 'Owns Boat'
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
}

# --- Common Coverages for easy reference ---
ALL_COVERAGES = {
    "PersonalAccident", "PassengerProtection", "EngineProtection",
    "ReturnToInvoice", "RoadsideAssistance", "NCBProtection",
    "KeyReplacement", "Consumables", "LegLiabPaidDriver",
    "UnnamedPARider", "ZeroDeprecCover"
}

# Define sets of coverages for each scenario
COVERAGES_SCENARIO_1 = {
    "ZeroDeprecCover", "EngineProtection", "ReturnToInvoice",
    "RoadsideAssistance", "Consumables", "PersonalAccident",
    "PassengerProtection"
} # Reason: "New car"

COVERAGES_SCENARIO_2 = {
    "EngineProtection", "Consumables", "PersonalAccident"
} # Reason: "Flood prone zones"

COVERAGES_SCENARIO_3 = {
    "ZeroDeprecCover", "EngineProtection", "ReturnToInvoice",
    "RoadsideAssistance", "Consumables", "PassengerProtection",
    "NCBProtection", "KeyReplacement", "LegLiabPaidDriver",
    "UnnamedPARider"
} # Reason: "Premium cars"

COVERAGES_DEFAULT = {
    "PersonalAccident", "RoadsideAssistance"
} # Reason: "Standard recommendations"


# --- API Endpoints ---

@app.get("/")
async def welcome():
    return {"message": "Welcome to recommendation system"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(data: InputData):
    if not loaded_models:
        raise HTTPException(status_code=503, detail="Prediction service unavailable: Model not loaded.")
    try:
        input_dict = data.dict(exclude_none=True)
        # Create a DataFrame with columns matching the training data
        mapped_input_data = {}
        for request_key, request_value in input_dict.items():
            if request_key in INPUT_FIELD_MAPPING:
                mapped_input_data[INPUT_FIELD_MAPPING[request_key]] = [request_value]
            elif request_key == 'Owns_Boat':
                mapped_input_data['Owns Boat'] = [request_value]

        input_df = pd.DataFrame(mapped_input_data)

        input_encoded = pd.get_dummies(input_df, columns=[INPUT_FIELD_MAPPING.get(col) for col in CATEGORICAL_FEATURES_FASTAPI if INPUT_FIELD_MAPPING.get(col) in input_df.columns or col == 'Owns Boat' and 'Owns Boat' in input_df.columns], drop_first=True)

        final_input = input_encoded.reindex(columns=loaded_feature_columns, fill_value=0)

        predictions = []

        for target_col, model in loaded_models.items():
            prediction = model.predict(final_input)[0]
            predicted_risk_score = int(np.clip(np.round(prediction), 1, 5))

            importance = getattr(model, 'feature_importances_', None)
            feature_importance_dict_full = {}
            if importance is not None:
                feature_importance_dict_full = {loaded_feature_columns[i]: importance[i] for i in range(len(loaded_feature_columns))}

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

            coverage_name = target_col.replace("Risk Score - ", "")
            prediction_item = PredictionItem(
                coverage=coverage_name,
                predicted_value=predicted_risk_score,
                factors=input_features_for_factors
            )
            predictions.append(prediction_item)

        return PredictionResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_coverage", response_model=CoverageRecommendationResponse)
async def predict_coverage(data: CoverageRequest):
    """
    Recommends vehicle coverages based strictly on the three original scenarios
    provided in the requirements, allowing for multiple entries if multiple
    original conditions are met.
    """
    current_year = datetime.datetime.now().year # Current year is 2025
    final_recommendations_list: List[CoverageRecommendationItem] = []
    
    # Normalize city input for case-insensitive matching
    city_lower = data.City.lower()
    
    # Define cities prone to floods
    flood_prone_cities = {"mumbai", "chennai", "kochi", "guwahati"}

    # Evaluate conditions
    is_new_car_condition = data.VehicleYear >= current_year - 1 # True if VehicleYear is 2024 or 2025
    is_flood_prone_city = city_lower in flood_prone_cities
    is_premium_car = data.VehicleIDV > 2500000

    # Apply each original scenario independently
    # If the condition is met, add its specific recommendation to the list.
    
    # Scenario 1: New Car
    if is_new_car_condition:
        final_recommendations_list.append(
            CoverageRecommendationItem(
                coveragesPatternCode=sorted(list(COVERAGES_SCENARIO_1)),
                reason="New car"
            )
        )
    
    # Scenario 2: Flood Prone City
    if is_flood_prone_city:
        final_recommendations_list.append(
            CoverageRecommendationItem(
                coveragesPatternCode=sorted(list(COVERAGES_SCENARIO_2)),
                reason="Flood prone zones"
            )
        )

    # Scenario 3: Premium Car
    if is_premium_car:
        final_recommendations_list.append(
            CoverageRecommendationItem(
                coveragesPatternCode=sorted(list(COVERAGES_SCENARIO_3)),
                reason="Premium cars"
            )
        )
            
    # Default scenario: If none of the above specific conditions were met
    if not final_recommendations_list:
        final_recommendations_list.append(
            CoverageRecommendationItem(
                coveragesPatternCode=sorted(list(COVERAGES_DEFAULT)),
                reason="Standard recommendations"
            )
        )
            
    return CoverageRecommendationResponse(recommendations=final_recommendations_list)


# Sample Request (to test with a client like Postman or curl) for /predict
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

# Sample Request for /predict_coverage - Example 1: New Car and Premium Car in Flood Prone City
# This will now return three separate recommendations, one for each original scenario met.
"""
{
    "Policy_ID": "publicId_123",
    "NewVehicle": "Yes",
    "VehicleType": "4wheeler",
    "VehicleMake": "Toyota",
    "VehicleModel": "Camry",
    "VehicleYear": 2024,
    "VehicleIDV": 2700000,
    "VehicleFuelType": "Petrol",
    "City": "Chennai",
    "PostalCode": 600001
}
"""

# Expected Output for Example 1 (All three original scenarios are met):
# {
#   "recommendations": [
#     {
#       "coveragesPatternCode": [
#         "Consumables",
#         "EngineProtection",
#         "PassengerProtection",
#         "PersonalAccident",
#         "ReturnToInvoice",
#         "RoadsideAssistance",
#         "ZeroDeprecCover"
#       ],
#       "reason": "New car"
#     },
#     {
#       "coveragesPatternCode": [
#         "Consumables",
#         "EngineProtection",
#         "PersonalAccident"
#       ],
#       "reason": "Flood prone zones"
#     },
#     {
#       "coveragesPatternCode": [
#         "Consumables",
#         "EngineProtection",
#         "KeyReplacement",
#         "LegLiabPaidDriver",
#         "NCBProtection",
#         "PassengerProtection",
#         "ReturnToInvoice",
#         "RoadsideAssistance",
#         "UnnamedPARider",
#         "ZeroDeprecCover"
#       ],
#       "reason": "Premium cars"
#     }
#   ]
# }


# Sample request for /predict_coverage - Example 2: only New Car
"""
{
    "Policy_ID": "publicId_456",
    "NewVehicle": "Yes",
    "VehicleType": "4wheeler",
    "VehicleMake": "Honda",
    "VehicleModel": "Civic",
    "VehicleYear": 2025,
    "VehicleIDV": 1500000,
    "VehicleFuelType": "Petrol",
    "City": "Bangalore",
    "PostalCode": 560001
}
"""
# Expected output for the above Sample Request 2:
# {
#   "recommendations": [
#     {
#       "coveragesPatternCode": [
#         "Consumables",
#         "EngineProtection",
#         "PassengerProtection",
#         "PersonalAccident",
#         "ReturnToInvoice",
#         "RoadsideAssistance",
#         "ZeroDeprecCover"
#       ],
#       "reason": "New car"
#     }
#   ]
# }

# Sample request for /predict_coverage - Example 3: only Flood Prone City
"""
{
    "Policy_ID": "publicId_789",
    "NewVehicle": "No",
    "VehicleType": "4wheeler",
    "VehicleMake": "Maruti",
    "VehicleModel": "Swift",
    "VehicleYear": 2020,
    "VehicleIDV": 500000,
    "VehicleFuelType": "Petrol",
    "City": "Kochi",
    "PostalCode": 682001
}
"""
# Expected output for the above Sample Request 3:
# {
#   "recommendations": [
#     {
#       "coveragesPatternCode": [
#         "Consumables",
#         "EngineProtection",
#         "PersonalAccident"
#       ],
#       "reason": "Flood prone zones"
#     }
#   ]
# }

# Sample request for /predict_coverage - Example 4: Default Scenario (no specific conditions met)
"""
{
    "Policy_ID": "publicId_101",
    "NewVehicle": "No",
    "VehicleType": "4wheeler",
    "VehicleMake": "Hyundai",
    "VehicleModel": "i20",
    "VehicleYear": 2022,
    "VehicleIDV": 700000,
    "VehicleFuelType": "Petrol",
    "City": "Pune",
    "PostalCode": 411001
}
"""
# Expected output for the above Sample Request 4:
# {
#   "recommendations": [
#     {
#       "coveragesPatternCode": [
#         "PersonalAccident",
#         "RoadsideAssistance"
#       ],
#       "reason": "Standard recommendations"
#     }
#   ]
# }