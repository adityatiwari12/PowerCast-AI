import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

class ProjectPredictor:
    def __init__(self):
        """Initialize the predictor with trained models"""
        self.models_dir = os.path.dirname(os.path.abspath(__file__))
        self.preprocessor = None
        self.models = {}
        self.explainers = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessor"""
        # Load preprocessor
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        else:
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        # Load models
        model_types = ['lr_cost', 'rf_cost', 'xgb_cost', 'lr_schedule', 'rf_schedule', 'xgb_schedule']
        for model_type in model_types:
            model_path = os.path.join(self.models_dir, f'{model_type}.pkl')
            if os.path.exists(model_path):
                self.models[model_type] = joblib.load(model_path)
            else:
                print(f"Warning: Model {model_type} not found at {model_path}")
        
        # Create explainers for XGBoost models
        if 'xgb_cost' in self.models:
            self.explainers['xgb_cost'] = shap.TreeExplainer(self.models['xgb_cost'])
        if 'xgb_schedule' in self.models:
            self.explainers['xgb_schedule'] = shap.TreeExplainer(self.models['xgb_schedule'])
    
    def preprocess_data(self, project_data):
        """Preprocess project data for prediction"""
        if isinstance(project_data, dict):
            # Convert single project dict to DataFrame
            project_data = pd.DataFrame([project_data])
        
        # Apply preprocessing
        if self.preprocessor:
            X = self.preprocessor.transform(project_data)
            return X
        else:
            raise ValueError("Preprocessor not loaded")
    
    def predict(self, project_data, model_type='xgb'):
        """Make predictions for project data"""
        # Preprocess data
        X = self.preprocess_data(project_data)
        
        # Make predictions
        results = {}
        
        # Cost overrun prediction
        cost_model = self.models.get(f'{model_type}_cost')
        if cost_model:
            results['cost_overrun_probability'] = cost_model.predict_proba(X)[:, 1]
            results['cost_overrun_prediction'] = cost_model.predict(X)
        
        # Schedule delay prediction
        schedule_model = self.models.get(f'{model_type}_schedule')
        if schedule_model:
            results['schedule_delay_probability'] = schedule_model.predict_proba(X)[:, 1]
            results['schedule_delay_prediction'] = schedule_model.predict(X)
        
        return results
    
    def explain_prediction(self, project_data, target_type='cost'):
        """Generate explanation for prediction"""
        # Preprocess data
        X = self.preprocess_data(project_data)
        
        # Get explainer
        explainer = self.explainers.get(f'xgb_{target_type}')
        if not explainer:
            raise ValueError(f"Explainer for {target_type} not available")
        
        # Generate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Get feature names if available
        feature_names = getattr(self.preprocessor, 'get_feature_names', lambda: None)()
        
        # Create explanation
        explanation = {}
        
        # Get absolute SHAP values
        abs_shap_values = np.abs(shap_values)
        
        # Get indices of top features
        top_indices = np.argsort(-abs_shap_values[0])[:5]  # Top 5 features
        
        # Create explanation dictionary
        explanation['top_features'] = []
        for idx in top_indices:
            feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
            impact = shap_values[0][idx]
            explanation['top_features'].append({
                'feature': feature_name,
                'impact': float(impact),
                'abs_impact': float(abs(impact)),
                'direction': 'Increases risk' if impact > 0 else 'Decreases risk'
            })
        
        # Calculate overall risk contribution percentages
        total_impact = np.sum(abs_shap_values[0])
        for feature in explanation['top_features']:
            feature['contribution_pct'] = (feature['abs_impact'] / total_impact) * 100 if total_impact > 0 else 0
        
        return explanation
    
    def generate_shap_plot(self, project_data, target_type='cost', output_path=None):
        """Generate and save SHAP plot for a prediction"""
        # Preprocess data
        X = self.preprocess_data(project_data)
        
        # Get explainer
        explainer = self.explainers.get(f'xgb_{target_type}')
        if not explainer:
            raise ValueError(f"Explainer for {target_type} not available")
        
        # Generate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Get feature names if available
        feature_names = getattr(self.preprocessor, 'get_feature_names', lambda: None)()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Force plot for single prediction
        if feature_names:
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                X[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        else:
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                X[0],
                matplotlib=True,
                show=False
            )
        
        # Save plot if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            return output_path
        else:
            return plt