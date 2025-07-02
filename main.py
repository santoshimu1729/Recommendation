from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Import Field for better validation/description
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set # Import Set for unique coverages
from sklearn.metrics import pairwise_distances
import datetime # To get the current year

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
    coveragesPatternCode: str = Field(..., description="Comma-separated list of recommended coverages")
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

# --- Common Coverages for easy reference ---
ALL_COVERAGES = {
    "PersonalAccident", "PassengerProtection", "EngineProtection",
    "ReturnToInvoice", "RoadsideAssistance", "NCBProtection",
    "KeyReplacement", "Consumables", "LegLiabPaidDriver",
    "UnnamedPARider", "ZeroDeprecCover"
}

# Define sets of coverages for each scenario to easily take unions
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
                mapped_input_data['Owns Boat'] = [request_value] # Handle 'Owns Boat' if provided

        input_df = pd.DataFrame(mapped_input_data)

        # Preprocess the input data
        # Ensure that get_dummies handles potential new categorical features not seen in training
        # For simplicity, this example assumes categories are consistent or handled by fill_value=0
        input_encoded = pd.get_dummies(input_df, columns=[INPUT_FIELD_MAPPING.get(col) for col in CATEGORICAL_FEATURES_FASTAPI if INPUT_FIELD_MAPPING.get(col) in input_df.columns or col == 'Owns Boat' and 'Owns Boat' in input_df.columns], drop_first=True)

        # Reindex to align with the training data's columns, filling missing with 0
        final_input = input_encoded.reindex(columns=loaded_feature_columns, fill_value=0)

        predictions = []

        for target_col, model in loaded_models.items():
            prediction = model.predict(final_input)[0]
            predicted_risk_score = int(np.clip(np.round(prediction), 1, 5))

            # Get global feature importance (this assumes model has feature_importances_)
            # If the model doesn't have it (e.g., some simple linear models), you'd need a different approach
            importance = getattr(model, 'feature_importances_', None)
            feature_importance_dict_full = {}
            if importance is not None:
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
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_coverage", response_model=CoverageRecommendationResponse)
async def predict_coverage(data: CoverageRequest):
    """
    Recommends vehicle coverages based on predefined scenarios.
    """
    current_year = datetime.datetime.now().year
    recommended_coverages: Set[str] = set() # Use a set to store unique coverages
    reasons: List[str] = []
    
    # Normalize city input for case-insensitive matching
    city_lower = data.City.lower()
    
    # Define cities prone to floods
    flood_prone_cities = {"mumbai", "chennai", "kochi", "guwahati"}

    # Evaluate scenarios in order of specificity (most specific first)

    # Scenario 7: All three conditions
    is_new_car_condition = data.VehicleYear >= current_year - 1 # Current year - 1 is 2024
    is_flood_prone_city = city_lower in flood_prone_cities
    is_premium_car = data.VehicleIDV > 2500000

    if is_new_car_condition and is_flood_prone_city and is_premium_car:
        recommended_coverages.update(COVERAGES_SCENARIO_1)
        recommended_coverages.update(COVERAGES_SCENARIO_2)
        recommended_coverages.update(COVERAGES_SCENARIO_3)
        reasons.append("New premium car in flood prone zone")
    # Combined Scenarios (more specific than individual ones)
    elif is_new_car_condition and is_flood_prone_city:
        recommended_coverages.update(COVERAGES_SCENARIO_1)
        recommended_coverages.update(COVERAGES_SCENARIO_2)
        reasons.append("New car in flood prone zone")
    elif is_new_car_condition and is_premium_car:
        recommended_coverages.update(COVERAGES_SCENARIO_1)
        recommended_coverages.update(COVERAGES_SCENARIO_3)
        reasons.append("New premium car")
    elif is_flood_prone_city and is_premium_car:
        recommended_coverages.update(COVERAGES_SCENARIO_2)
        recommended_coverages.update(COVERAGES_SCENARIO_3)
        reasons.append("Premium car in flood prone zone")
    # Original Scenarios (if no combined scenario matches)
    elif is_new_car_condition:
        recommended_coverages.update(COVERAGES_SCENARIO_1)
        reasons.append("New car")
    elif is_flood_prone_city:
        recommended_coverages.update(COVERAGES_SCENARIO_2)
        reasons.append("Flood prone zones")
    elif is_premium_car:
        recommended_coverages.update(COVERAGES_SCENARIO_3)
        reasons.append("Premium cars")
    else:
        # Default scenario if no specific conditions are met
        recommended_coverages.update(COVERAGES_DEFAULT)
        reasons.append("Standard recommendations")

    # Format the response
    return CoverageRecommendationResponse(
        recommendations=[
            CoverageRecommendationItem(
                coveragesPatternCode=", ".join(sorted(list(recommended_coverages))), # Sort for consistent output
                reason="; ".join(reasons) # Combine reasons if multiple apply
            )
        ]
    )


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

# Sample Request for /predict_coverage
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

# Sample request for /predict_coverage (another example: only New Car)
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

# Sample request for /predict_coverage (another example: only Flood Prone City)
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

# Sample request for /predict_coverage (another example: Default)
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