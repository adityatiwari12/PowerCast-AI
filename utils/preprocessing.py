import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
import os

# Download NLTK resources if not already downloaded
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

class DataPreprocessor:
    def __init__(self):
        download_nltk_resources()
        self.numeric_features = [
            'planned_duration_months', 'planned_budget_millions',
            'elevation_meters', 'soil_quality', 'avg_rainfall_mm',
            'avg_temperature_celsius', 'material_cost_index',
            'labor_availability_index', 'vendor_reliability_score',
            'vendor_past_defaults', 'permit_time_months', 'regulatory_complexity'
        ]
        
        self.categorical_features = [
            'project_type', 'terrain_type', 'monsoon_affected'
        ]
        
        self.text_features = ['delay_description']
        
        self.date_features = ['start_date', 'planned_end_date']
        
        self.preprocessor = None
        self.text_vectorizer = None
        self.date_encoder = None
    
    def extract_text_features(self, texts):
        """Extract features from text descriptions"""
        # Simple keyword extraction
        stop_words = set(stopwords.words('english'))
        
        # Keywords that might indicate issues
        risk_keywords = {
            'delay': 1.5, 'issue': 1.2, 'problem': 1.2, 'challenge': 1.1,
            'weather': 1.3, 'rain': 1.3, 'storm': 1.4, 'flood': 1.5,
            'vendor': 1.4, 'supplier': 1.4, 'delivery': 1.3,
            'permit': 1.3, 'approval': 1.3, 'regulation': 1.2,
            'design': 1.2, 'technical': 1.1, 'specification': 1.1,
            'labor': 1.3, 'worker': 1.3, 'skill': 1.2, 'shortage': 1.4,
            'material': 1.3, 'quality': 1.2, 'equipment': 1.3,
            'community': 1.2, 'protest': 1.4, 'opposition': 1.3,
            'bankruptcy': 1.5, 'financial': 1.3, 'budget': 1.2
        }
        
        features = []
        
        for text in texts:
            if pd.isna(text) or text == "":
                features.append([0, 0, 0])  # No risk for empty descriptions
                continue
                
            # Tokenize and clean
            words = re.findall(r'\b\w+\b', text.lower())
            words = [w for w in words if w not in stop_words and len(w) > 2]
            
            # Calculate risk score based on keywords
            risk_score = 0
            matched_keywords = 0
            
            for word in words:
                if word in risk_keywords:
                    risk_score += risk_keywords[word]
                    matched_keywords += 1
            
            # Normalize risk score
            if matched_keywords > 0:
                risk_score = risk_score / matched_keywords
            
            # Count number of issues mentioned
            issue_count = sum(1 for word in words if word in risk_keywords)
            
            # Create feature vector [risk_score, keyword_count, text_length]
            features.append([
                risk_score,
                issue_count,
                len(words)
            ])
        
        return np.array(features)
    
    def extract_date_features(self, df):
        """Extract features from date columns"""
        date_df = pd.DataFrame()
        
        for col in self.date_features:
            if col in df.columns:
                # Convert to datetime
                dates = pd.to_datetime(df[col])
                
                # Extract features
                date_df[f'{col}_month'] = dates.dt.month
                date_df[f'{col}_quarter'] = dates.dt.quarter
                date_df[f'{col}_year'] = dates.dt.year
                
                # Is monsoon season? (assuming June-September is monsoon)
                date_df[f'{col}_is_monsoon'] = dates.dt.month.isin([6, 7, 8, 9]).astype(int)
        
        return date_df
    
    def fit(self, df):
        """Fit the preprocessing pipeline to the data"""
        # Create preprocessing pipeline for numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessing pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(df)
        
        return self
    
    def transform(self, df):
        """Transform the data using the fitted pipeline"""
        # Process structured data
        X_structured = self.preprocessor.transform(df)
        
        # Process text data if available
        if any(col in df.columns for col in self.text_features):
            text_data = df.get('delay_description', pd.Series([''] * len(df)))
            X_text = self.extract_text_features(text_data)
            
            # Convert sparse matrix to dense if needed
            if hasattr(X_structured, 'toarray'):
                X_structured = X_structured.toarray()
            
            # Combine structured and text features
            X = np.hstack([X_structured, X_text])
        else:
            X = X_structured
        
        # Process date features if available
        if any(col in df.columns for col in self.date_features):
            date_features = self.extract_date_features(df)
            
            # Convert sparse matrix to dense if needed
            if hasattr(X, 'toarray'):
                X = X.toarray()
            
            # Combine with date features
            X = np.hstack([X, date_features.values])
        
        return X
    
    def fit_transform(self, df):
        """Fit and transform the data"""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self):
        """Get the names of the transformed features"""
        feature_names = []
        
        # Get names from column transformer
        if self.preprocessor is not None:
            ct_feature_names = []
            for name, _, columns in self.preprocessor.transformers_:
                if name == 'num':
                    ct_feature_names.extend(columns)
                elif name == 'cat':
                    # Get one-hot encoded feature names
                    encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                    for i, col in enumerate(columns):
                        cats = encoder.categories_[i]
                        ct_feature_names.extend([f'{col}_{cat}' for cat in cats])
            
            feature_names.extend(ct_feature_names)
        
        # Add text feature names
        feature_names.extend(['text_risk_score', 'text_issue_count', 'text_length'])
        
        # Add date feature names
        for col in self.date_features:
            feature_names.extend([
                f'{col}_month', f'{col}_quarter', f'{col}_year', f'{col}_is_monsoon'
            ])
        
        return feature_names