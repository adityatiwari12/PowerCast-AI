# PowerCast AI - Usage Guide

## Quick Start

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```
   python run.py
   ```
   This will start the Streamlit dashboard and open it in your default browser.

## Using the Dashboard

The PowerCast AI dashboard has three main sections:

### 1. Dashboard Overview

- Displays summary metrics of all projects
- Shows project distribution by type
- Presents risk analysis by project type
- Lists all projects with their risk scores

### 2. Project Analysis

- Select a specific project to analyze in detail
- View project details and environmental factors
- See cost overrun and schedule delay risk analysis
- Explore top risk factors with their impact values
- Get tailored recommendations to mitigate risks

### 3. New Project Prediction

- Enter details for a new project
- Get instant risk predictions for cost overruns and schedule delays
- View explanations of which factors contribute most to risk
- See recommendations for risk mitigation
- Save the project for future reference

## Using the API

PowerCast AI also provides a REST API for integration with other systems:

1. **Start the API Server**:
   ```
   cd app
   uvicorn api:app --reload
   ```

2. **API Endpoints**:

   - `GET /`: Welcome message
   - `POST /predict`: Predict risks for a new project

3. **Example API Request**:
   ```python
   import requests
   import json

   url = "http://localhost:8000/predict"
   
   project_data = {
       "project_type": "Substation",
       "terrain_type": "Urban",
       "planned_duration_months": 24,
       "planned_budget_millions": 25.0,
       "elevation_meters": 500,
       "soil_quality": 7.5,
       "avg_rainfall_mm": 1200,
       "avg_temperature_celsius": 25,
       "monsoon_affected": 1,
       "material_cost_index": 6.5,
       "labor_availability_index": 4.5,
       "vendor_reliability_score": 6.0,
       "vendor_past_defaults": 2,
       "permit_time_months": 8,
       "regulatory_complexity": 7.0
   }
   
   response = requests.post(url, json=project_data)
   results = response.json()
   
   print(f"Cost Overrun Probability: {results['cost_overrun_probability']}")
   print(f"Schedule Delay Probability: {results['schedule_delay_probability']}")
   ```

## Testing the System

Run the test script to verify all components are working correctly:

```
python test_system.py
```

This will test:
- Data generation
- Preprocessing
- Model training
- Prediction functionality
- API endpoints

## Customization

### Adding New Features

To add new features to the model:

1. Update the `generate_data.py` file to include the new feature
2. Modify the `preprocessing.py` file to handle the new feature
3. Retrain the models using `train_models.py`

### Improving Models

To improve model performance:

1. Edit `train_models.py` to try different algorithms or hyperparameters
2. Run the training process again to generate new models
3. Test the new models using the test script

## Troubleshooting

- **Missing dependencies**: Ensure all packages in `requirements.txt` are installed
- **Model loading errors**: Check that model files exist in the `models` directory
- **API connection issues**: Verify the API server is running on the correct port
- **Dashboard not loading**: Check Streamlit installation and port availability