import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self, data_path=None):
        """Initialize the model trainer"""
        self.data_path = data_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                 'data', 'project_data.csv')
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.feature_names = None
        self.explainers = {}
        
    def load_data(self):
        """Load the project data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        return pd.read_csv(self.data_path)
    
    def prepare_data(self, df):
        """Prepare data for model training"""
        # Define target variables
        y_cost = df['cost_overrun']
        y_schedule = df['schedule_delay']
        
        # Preprocess features
        X = self.preprocessor.fit_transform(df)
        self.feature_names = self.preprocessor.get_feature_names()
        
        # Split data
        X_train, X_test, y_cost_train, y_cost_test, y_schedule_train, y_schedule_test = train_test_split(
            X, y_cost, y_schedule, test_size=0.2, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_cost_train': y_cost_train,
            'y_cost_test': y_cost_test,
            'y_schedule_train': y_schedule_train,
            'y_schedule_test': y_schedule_test
        }
    
    def train_models(self):
        """Train models for cost and schedule prediction"""
        # Load and prepare data
        df = self.load_data()
        data = self.prepare_data(df)
        
        # Train cost overrun models
        print("Training cost overrun models...")
        
        # Logistic Regression
        lr_cost = LogisticRegression(max_iter=1000, random_state=42)
        lr_cost.fit(data['X_train'], data['y_cost_train'])
        self.models['lr_cost'] = lr_cost
        
        # Random Forest
        rf_cost = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_cost.fit(data['X_train'], data['y_cost_train'])
        self.models['rf_cost'] = rf_cost
        
        # XGBoost
        xgb_cost = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_cost.fit(data['X_train'], data['y_cost_train'])
        self.models['xgb_cost'] = xgb_cost
        
        # Train schedule delay models
        print("Training schedule delay models...")
        
        # Logistic Regression
        lr_schedule = LogisticRegression(max_iter=1000, random_state=42)
        lr_schedule.fit(data['X_train'], data['y_schedule_train'])
        self.models['lr_schedule'] = lr_schedule
        
        # Random Forest
        rf_schedule = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_schedule.fit(data['X_train'], data['y_schedule_train'])
        self.models['rf_schedule'] = rf_schedule
        
        # XGBoost
        xgb_schedule = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_schedule.fit(data['X_train'], data['y_schedule_train'])
        self.models['xgb_schedule'] = xgb_schedule
        
        # Evaluate models
        self.evaluate_models(data)
        
        # Create model explainers
        self.create_explainers(data)
        
        return self.models
    
    def evaluate_models(self, data):
        """Evaluate trained models"""
        results = {}
        
        for model_name, model in self.models.items():
            # Determine which target to use based on model name
            if 'cost' in model_name:
                y_test = data['y_cost_test']
                target_type = 'cost'
            else:
                y_test = data['y_schedule_test']
                target_type = 'schedule'
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            results[model_name] = metrics
            
            print(f"\nEvaluation for {model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        return results
    
    def create_explainers(self, data):
        """Create SHAP explainers for the models"""
        # Create explainers for the best models (XGBoost)
        print("\nCreating model explainers...")
        
        # Cost overrun explainer
        explainer_cost = shap.TreeExplainer(self.models['xgb_cost'])
        self.explainers['xgb_cost'] = explainer_cost
        
        # Schedule delay explainer
        explainer_schedule = shap.TreeExplainer(self.models['xgb_schedule'])
        self.explainers['xgb_schedule'] = explainer_schedule
        
        # Generate and save example SHAP values
        shap_values_cost = explainer_cost.shap_values(data['X_test'])
        shap_values_schedule = explainer_schedule.shap_values(data['X_test'])
        
        # Save SHAP values for dashboard
        self.save_shap_summary_plot(explainer_cost, data['X_test'], 'cost')
        self.save_shap_summary_plot(explainer_schedule, data['X_test'], 'schedule')
    
    def save_shap_summary_plot(self, explainer, X_test, target_type):
        """Save SHAP summary plot to file"""
        plt.figure(figsize=(10, 8))
        shap_values = explainer.shap_values(X_test)
        
        # If feature names are available, use them
        if self.feature_names:
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                             show=False, plot_size=(10, 8))
        else:
            shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 8))
        
        # Create directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'static')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, f'shap_summary_{target_type}.png'), bbox_inches='tight', dpi=150)
        plt.close()
    
    def save_models(self):
        """Save trained models and preprocessor"""
        # Create models directory if it doesn't exist
        models_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(models_dir, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(models_dir, f'{model_name}.pkl'))
        
        print(f"Models and preprocessor saved to {models_dir}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_models()
    trainer.save_models()