"""
Model Evaluation Module for Reddit NLP Pipeline
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles evaluation of trained models."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.evaluation_results = {}
        logger.info("Model evaluator initialized")
    
    def evaluate_single_model(self, model_name: str, model, X_test: np.ndarray, y_test: np.ndarray, label_encoder) -> Dict[str, Any]:
        """Evaluate a single model."""
        print(f"Evaluating {model_name}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Classification report
            target_names = label_encoder.classes_
            class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            result = {
                'test_accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {str(e)}")
            return None
    
    def evaluate_all_models(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all trained models."""
        trained_models = models_data['models']
        data_splits = models_data['data_splits']
        label_encoder = models_data['label_encoder']
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        model_scores = {}
        
        for model_name, model_data in trained_models.items():
            model = model_data['model']
            result = self.evaluate_single_model(model_name, model, X_test, y_test, label_encoder)
            
            if result is not None:
                # Add training CV scores
                result['cv_mean'] = model_data['cv_mean']
                result['cv_std'] = model_data['cv_std']
                model_scores[model_name] = result
        
        # Find best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['test_accuracy'])
        best_accuracy = model_scores[best_model_name]['test_accuracy']
        
        logger.info(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return {
            'model_scores': model_scores,
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'label_encoder': label_encoder
        }
    
    def print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        print("\nMODEL EVALUATION SUMMARY")
        print("=" * 50)
        
        for model_name, scores in evaluation_results['model_scores'].items():
            print(f"{model_name:20} | Accuracy: {scores['test_accuracy']:.4f} | F1: {scores['f1_score']:.4f}")
        
        print(f"\nBest Model: {evaluation_results['best_model']} (Accuracy: {evaluation_results['best_accuracy']:.4f})")