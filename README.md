# PowerCast AI

A machine learning system that predicts project cost overruns and schedule delays for PowerGrid infrastructure projects.

## Features

- Ingests project-related data (historical + current)
- Trains ML models to predict probability of cost/time overruns
- Identifies key risk drivers (hotspots)
- Outputs results in a simple dashboard or visualization

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate synthetic data:
   ```
   python data/generate_data.py
   ```

3. Run the Streamlit dashboard:
   ```
   streamlit run app/dashboard.py
   ```

## Project Structure

- `data/`: Data generation and storage
- `app/`: Application code including dashboard
- `models/`: ML model training and prediction
- `utils/`: Utility functions
- `tests/`: Test files