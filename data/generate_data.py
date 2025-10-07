import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Ensure data directory exists
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

def generate_synthetic_data(n_samples=200):
    """
    Generate synthetic data for PowerGrid infrastructure projects
    """
    np.random.seed(42)
    
    # Project types and their characteristics
    project_types = ['Substation', 'Overhead Line', 'Underground Cable']
    project_type_probs = [0.4, 0.35, 0.25]
    
    # Terrain types
    terrain_types = ['Urban', 'Rural', 'Forest', 'Coastal', 'Mountain']
    
    # Generate basic project data
    data = {
        'project_id': [f'PRJ-{i:04d}' for i in range(1, n_samples + 1)],
        'project_type': np.random.choice(project_types, size=n_samples, p=project_type_probs),
        'start_date': [datetime.now() - timedelta(days=np.random.randint(30, 1000)) for _ in range(n_samples)],
        'planned_duration_months': np.random.randint(6, 36, size=n_samples),
        'planned_budget_millions': np.random.uniform(1, 100, size=n_samples).round(2),
        'terrain_type': np.random.choice(terrain_types, size=n_samples),
        'elevation_meters': np.random.randint(0, 2000, size=n_samples),
        'soil_quality': np.random.uniform(1, 10, size=n_samples).round(1),
        'avg_rainfall_mm': np.random.uniform(500, 2000, size=n_samples).round(1),
        'avg_temperature_celsius': np.random.uniform(-5, 35, size=n_samples).round(1),
        'monsoon_affected': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'material_cost_index': np.random.uniform(0.8, 1.5, size=n_samples).round(2),
        'labor_availability_index': np.random.uniform(0.6, 1.2, size=n_samples).round(2),
        'vendor_reliability_score': np.random.uniform(1, 10, size=n_samples).round(1),
        'vendor_past_defaults': np.random.randint(0, 5, size=n_samples),
        'permit_time_months': np.random.randint(1, 12, size=n_samples),
        'regulatory_complexity': np.random.uniform(1, 10, size=n_samples).round(1),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlations to make data more realistic
    
    # Larger projects tend to have higher budgets and longer durations
    for i, project_type in enumerate(project_types):
        mask = df['project_type'] == project_type
        if project_type == 'Substation':
            df.loc[mask, 'planned_budget_millions'] *= np.random.uniform(1.2, 1.5, size=mask.sum())
            df.loc[mask, 'planned_duration_months'] += np.random.randint(3, 8, size=mask.sum())
        elif project_type == 'Overhead Line':
            df.loc[mask, 'planned_budget_millions'] *= np.random.uniform(0.8, 1.2, size=mask.sum())
        
    # Terrain affects costs and timelines
    for terrain in terrain_types:
        mask = df['terrain_type'] == terrain
        if terrain == 'Mountain':
            df.loc[mask, 'planned_budget_millions'] *= np.random.uniform(1.1, 1.3, size=mask.sum())
            df.loc[mask, 'planned_duration_months'] += np.random.randint(2, 5, size=mask.sum())
        elif terrain == 'Forest':
            df.loc[mask, 'planned_budget_millions'] *= np.random.uniform(1.05, 1.2, size=mask.sum())
            df.loc[mask, 'planned_duration_months'] += np.random.randint(1, 4, size=mask.sum())
    
    # Generate target variables (actual outcomes)
    # Cost overrun factor (1.0 means no overrun, 1.2 means 20% overrun)
    base_cost_overrun = np.random.normal(1.1, 0.2, size=n_samples)
    
    # Factors that influence cost overrun
    cost_overrun = base_cost_overrun.copy()
    cost_overrun += (10 - df['vendor_reliability_score']) * 0.02  # Less reliable vendors cause more overruns
    cost_overrun += df['vendor_past_defaults'] * 0.03  # Past defaults increase overruns
    cost_overrun += df['regulatory_complexity'] * 0.01  # Regulatory complexity increases overruns
    cost_overrun += (df['monsoon_affected'] * 0.1)  # Monsoon affected projects have more overruns
    cost_overrun += (df['permit_time_months'] > 6) * 0.1  # Long permit times increase overruns
    
    # Schedule delay factor (1.0 means no delay, 1.5 means 50% longer than planned)
    base_schedule_delay = np.random.normal(1.15, 0.25, size=n_samples)
    
    # Factors that influence schedule delay
    schedule_delay = base_schedule_delay.copy()
    schedule_delay += (10 - df['vendor_reliability_score']) * 0.03
    schedule_delay += df['vendor_past_defaults'] * 0.04
    schedule_delay += (df['labor_availability_index'] < 0.8) * 0.15
    schedule_delay += (df['monsoon_affected'] * 0.2)
    schedule_delay += (df['permit_time_months'] > 6) * 0.15
    
    # Ensure minimum values of 1.0 (no overrun/delay) for some projects
    cost_overrun = np.maximum(cost_overrun, 0.95)
    schedule_delay = np.maximum(schedule_delay, 0.95)
    
    # Add some randomness to make it more realistic
    cost_overrun *= np.random.normal(1, 0.05, size=n_samples)
    schedule_delay *= np.random.normal(1, 0.05, size=n_samples)
    
    # Add target variables to dataframe
    df['actual_budget_millions'] = (df['planned_budget_millions'] * cost_overrun).round(2)
    df['actual_duration_months'] = (df['planned_duration_months'] * schedule_delay).round(0).astype(int)
    
    # Create binary target variables
    df['cost_overrun'] = (df['actual_budget_millions'] > df['planned_budget_millions'] * 1.05).astype(int)
    df['schedule_delay'] = (df['actual_duration_months'] > df['planned_duration_months'] * 1.05).astype(int)
    
    # Add text descriptions for some projects
    delay_reasons = [
        "Vendor delivery issues caused significant delays.",
        "Unexpected terrain challenges required design modifications.",
        "Permit approvals took longer than anticipated.",
        "Severe weather conditions halted construction for weeks.",
        "Labor shortages in key skills affected progress.",
        "Material quality issues required replacements.",
        "Local community protests delayed site access.",
        "Technical design changes were needed mid-project.",
        "Subcontractor bankruptcy caused disruption.",
        "Equipment failure required replacement parts."
    ]
    
    df['delay_description'] = ""
    delayed_projects = df[df['schedule_delay'] == 1].index
    df.loc[delayed_projects, 'delay_description'] = np.random.choice(delay_reasons, size=len(delayed_projects))
    
    # Convert dates to string for easier handling
    df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
    
    # Calculate end dates
    df['planned_end_date'] = pd.to_datetime(df['start_date']) + pd.to_timedelta(df['planned_duration_months'] * 30, unit='D')
    df['actual_end_date'] = pd.to_datetime(df['start_date']) + pd.to_timedelta(df['actual_duration_months'] * 30, unit='D')
    
    # Convert back to string
    df['planned_end_date'] = df['planned_end_date'].dt.strftime('%Y-%m-%d')
    df['actual_end_date'] = df['actual_end_date'].dt.strftime('%Y-%m-%d')
    
    return df

def save_data(df, filename='project_data.csv'):
    """Save the generated data to a CSV file"""
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    return filepath

if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_data(n_samples=200)
    
    # Save to CSV
    save_data(df)
    
    # Display sample
    print("\nSample data:")
    print(df.head())
    
    # Display summary statistics
    print("\nSummary statistics:")
    print(f"Total projects: {len(df)}")
    print(f"Projects with cost overrun: {df['cost_overrun'].sum()} ({df['cost_overrun'].mean()*100:.1f}%)")
    print(f"Projects with schedule delay: {df['schedule_delay'].sum()} ({df['schedule_delay'].mean()*100:.1f}%)")
    print(f"Average planned budget: ${df['planned_budget_millions'].mean():.2f}M")
    print(f"Average actual budget: ${df['actual_budget_millions'].mean():.2f}M")
    print(f"Average budget overrun: {(df['actual_budget_millions'].sum() / df['planned_budget_millions'].sum() - 1) * 100:.1f}%")