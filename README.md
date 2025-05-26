# Recommendation
RecommendationModel

This detailed business requirement document outlines the logic for recommending upsell coverages and cross-sell products in the US insurance market, utilizing an XGBoost model deployed via FastAPI. It also includes a Python script to generate skewed mock data for model training, reflecting the defined business logic.
Business Requirements Document: Insurance Coverage Recommendation Model (US Market)
1. Introduction
This document details the business requirements for an AI/ML model designed to recommend relevant upsell insurance coverages and cross-sell products to customers in the US market. The model will leverage an XGBoost algorithm and be deployed as a FastAPI application, aiming to enhance customer satisfaction, increase average policy value, and optimize product offering.
2. Objectives
Increase Upsell Conversion: Proactively recommend relevant add-on coverages (Zero Depreciation, Engine Protection, Roadside Assistance, Return to Invoice, Bodily Injury) to enhance policy value and customer protection.
Boost Cross-sell Adoption: Identify opportunities to cross-sell related insurance products (Personal Accident Coverage, Travel Insurance, Boat Insurance) based on customer profiles and inferred needs.
Improve Customer Experience: Provide personalized recommendations that align with customer risk profiles and likely needs, leading to greater satisfaction.
Optimize Product Strategy: Gain insights into customer segments and their receptiveness to various coverages and products.
3. Scope
The scope of this project includes:
Defining input parameters obtainable from users.
Defining output parameters (risk scores for each coverage/product).
Establishing business logic for each risk score prediction, specifically tailored to the US market.
Developing an XGBoost model for risk score prediction.
Deploying the model as a FastAPI application.
Generating skewed mock data for model training purposes.
4. Input Parameters
The following input parameters will be collected from users:
Age: Numerical (e.g., 18-99)
Gender: Categorical (Male, Female, Non-binary, Prefer not to say)
Marital Status: Categorical (Single, Married, Divorced, Widowed)
Profession/Occupation: Categorical (e.g., Office Professional, Healthcare, Tradesperson, Student, Retired, etc. - a broader classification may be needed initially)
Driving Experience (in years): Numerical (e.g., 0-80)
Past Claim History (Number of claims in past 5 years): Numerical (e.g., 0-10+)
Past Traffic Violations (Number of violations in past 5 years): Numerical (e.g., 0-10+)
Vehicle Age (in years): Numerical (e.g., 0-30)
Vehicle Type/Category: Categorical (Sedan, SUV, Hatchback, Pickup Truck, Minivan, Sports Car, Motorcycle, Electric Vehicle)
Vehicle Make: Categorical (e.g., Ford, Chevrolet, Toyota, Honda, BMW, Tesla, etc.)
Vehicle Model: Categorical (e.g., F-150, Silverado, Camry, Civic, 3 Series, Model 3, etc.)
Vehicle Value/Price: Numerical (e.g., $5,000 - $250,000+)
Primary Use of Vehicle: Categorical (Commuting, Personal Use, Business Use, Pleasure)
Estimated Annual Mileage: Numerical (e.g., 0-50,000+)
Vehicle Safety Features: Binary (e.g., ABS, Airbags, Electronic Stability Control, Blind Spot Monitoring, Lane Keeping Assist, Automatic Emergency Braking - Yes/No for each)
Parking Location: Categorical (Secured Garage, Private Driveway, Street Parking, Public Lot)
Geographic Location/City: Categorical (e.g., New York, Los Angeles, Chicago, Houston, Phoenix, etc. - State and Zip Code might be more granular for risk assessment)
Owns Boat: Binary (Yes/No - Crucial for Boat Insurance)
5. Output Parameters (Risk Scores)
Each output parameter will be an ordinal risk score on a scale of 1-5, where 1 indicates very low risk/need and 5 indicates very high risk/need.
Risk Score - Zero Depreciation: Likelihood of significant vehicle depreciation leading to financial loss in case of total loss or major damage.
Risk Score - Engine Protection: Likelihood of engine damage due to external factors not typically covered by standard comprehensive insurance (e.g., waterlogging, oil leakage, hydrostatic lock).
Risk Score - Road Assistance: Likelihood of needing roadside assistance services (e.g., towing, flat tire change, battery jump-start, fuel delivery).
Risk Score - Return to Invoice: Likelihood of total loss or theft where the current market value is significantly less than the original invoice value.
Risk Score - Bodily Injury (Third Party): Likelihood of the insured being at fault in an accident causing injury to a third party.
Risk Score - Personal Accident Coverage (for the insured): Likelihood of the insured suffering personal injury or death due to a vehicle accident.
Risk Score - Travel Insurance: Likelihood of the customer traveling frequently or internationally, where travel insurance would be beneficial.
Risk Score - Boat Insurance: Relevance of boat insurance (only applicable if the customer owns a boat).
6. Business Logic for Risk Score Prediction (US Market Context)
The following logic will guide the model's behavior and the generation of mock data. This logic is based on general insurance principles and common risk factors in the US.
6.1. Upsell Coverages
Risk Score - Zero Depreciation:


High Risk (4-5): New vehicles (Vehicle Age 0-3 years), High Vehicle Value, Luxury/Sports Cars, Primary Use: Personal/Pleasure (where owners are more likely to maintain value). Drivers with clean records (fewer claims/violations) might be more attuned to maintaining vehicle value.
Medium Risk (2-3): Mid-range vehicle age (4-7 years), average vehicle value.
Low Risk (1): Older vehicles (8+ years), low vehicle value.
US Specific: Higher depreciation rates for certain makes/models (e.g., some luxury cars, EVs) could slightly increase this score.
Risk Score - Engine Protection:


High Risk (4-5): Newer vehicles (Vehicle Age 0-5 years) (as engine repairs are more costly), High Vehicle Value, Luxury/Sports Cars, Vehicles prone to specific engine issues (some European makes, certain models with known engine problems). Driving in areas prone to floods (e.g., coastal cities, hurricane-prone regions).
Medium Risk (2-3): Mid-range vehicle age (6-10 years), average vehicle value.
Low Risk (1): Older vehicles (10+ years), low vehicle value, or vehicles with simple, robust engines.
US Specific: Focus on specific urban areas known for flash floods or heavy rain (e.g., parts of Florida, Gulf Coast states, Midwest during spring storms).
Risk Score - Road Assistance:


High Risk (4-5): Older vehicles (Vehicle Age 10+ years), High Annual Mileage, Lower Vehicle Safety Features (implying older tech/reliability issues), Profession involving extensive driving (e.g., Sales, Delivery), Younger/Inexperienced Drivers (more prone to minor mishaps).
Medium Risk (2-3): Mid-range vehicle age (5-9 years), average mileage.
Low Risk (1): New vehicles (0-4 years), low annual mileage, high safety features.
US Specific: Commuters in congested cities might value this more. Consider road conditions in certain states (e.g., rough roads in the Northeast).
Risk Score - Return to Invoice:


High Risk (4-5): New vehicles (Vehicle Age 0-2 years), High Vehicle Value, Luxury/Sports Cars, High Theft Rate Makes/Models (e.g., certain Ford trucks, Honda Civics, Hyundai/Kia models targeted for theft in some regions), Parking Location: Street Parking/Public Lot.
Medium Risk (2-3): Mid-range vehicle age (3-5 years), average vehicle value.
Low Risk (1): Older vehicles (6+ years), low vehicle value, low theft risk models, Secured Garage Parking.
US Specific: This is highly relevant for new car purchases in the US, especially with increasing vehicle prices and common gap insurance offerings. Theft rates vary significantly by city/state.
Risk Score - Bodily Injury (Third Party):


High Risk (4-5): Younger Drivers (Age 18-25), Inexperienced Drivers (0-5 years), Past Claim History (especially at-fault accidents), Past Traffic Violations (especially reckless driving, DUI), Sports Cars/High-Performance Vehicles, High Annual Mileage, Primary Use: Business Use.
Medium Risk (2-3): Mid-age drivers (26-55), average experience, moderate claims/violations.
Low Risk (1): Older, experienced drivers (55+ and 20+ years driving experience), clean record, low mileage, personal use.
US Specific: States with high population density and traffic congestion (e.g., California, New York, Florida, Texas) tend to have higher BI claims. Specific occupations involving frequent driving increase risk.
6.2. Cross-Sell Products
Risk Score - Personal Accident Coverage (for the insured):


High Risk (4-5): Younger Drivers (Age 18-30), Inexperienced Drivers, High Annual Mileage, Profession involving extensive driving or higher physical risk, Primary Use: Business Use, Sports Car/Motorcycle.
Medium Risk (2-3): Average age and experience, moderate mileage.
Low Risk (1): Older, experienced drivers, low mileage, personal use, low-risk vehicle type.
US Specific: This coverage is often bundled or a low-cost add-on in the US. Consider higher rates for drivers in states with higher accident fatalities.
Risk Score - Travel Insurance:


High Risk (4-5): Professions implying frequent travel (e.g., Consultants, Sales, Executives), High Vehicle Value (suggesting higher income/lifestyle), Younger/Middle-aged (25-55) demographic, Urban Geographic Location (proximity to major airports).
Medium Risk (2-3): General population with occasional travel.
Low Risk (1): Students, Retired individuals with limited travel, low income, rural locations.
US Specific: Focus on individuals who likely travel for business or leisure. No direct vehicle link, but inferred from lifestyle/profession/affluence.
Risk Score - Boat Insurance:


High Risk (5): Only if "Owns Boat" is 'Yes'. All other factors are secondary.
Low Risk (1): Only if "Owns Boat" is 'No'.
US Specific: Heavily dependent on whether the user lives near coastal areas or major lakes/rivers (e.g., Florida, Great Lakes states, Pacific Northwest, Gulf Coast). If they own a boat, the risk is automatically high.
7. Model Training and Deployment
Algorithm: XGBoost (eXtreme Gradient Boosting) will be used for its proven performance in tabular data and ability to handle various data types. Separate models can be trained for each output risk score, or a multi-output model can be explored if computationally feasible and beneficial.
Deployment: The trained models will be deployed as a FastAPI application, providing a lightweight and high-performance API for real-time predictions.
Training Data: The generated mock data (as per Section 8) will be used for initial model training. Ongoing data collection from the deployed system will be crucial for continuous model improvement.

