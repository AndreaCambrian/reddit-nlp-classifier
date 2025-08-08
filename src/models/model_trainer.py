"""
Model Training Module for Reddit NLP Pipeline
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training of multiple ML models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.label_encoder = LabelEncoder()
        logger.info("Model trainer initialized")
    
    def prepare_data(self, features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        X = features['X']
        y = features['y']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, model_name: str, model, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train a single model."""
        print(f"Training {model_name}...")
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            result = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{model_name} trained successfully (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            return None
    
    def train_all_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Train all models."""
        X_train, X_test, y_train, y_test = self.prepare_data(features)
        
        # Define models
        model_configs = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50))
        }
        
        # Train each model
        trained_models = {}
        
        for model_name, model in model_configs.items():
            result = self.train_single_model(model_name, model, X_train, y_train)
            if result is not None:
                trained_models[model_name] = result
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        
        return {
            'models': trained_models,
            'data_splits': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            },
            'label_encoder': self.label_encoder
        }