"""
Visualization Module for Reddit NLP Pipeline
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ResultPlotter:
    """Handles all visualization and plotting operations."""
    
    def __init__(self):
        """Initialize plotter."""
        plt.style.use('default')
        sns.set_palette("husl")
        logger.info("Result plotter initialized")
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Any], save_dir: Path):
        """Plot model comparison chart."""
        model_scores = evaluation_results['model_scores']
        
        models = list(model_scores.keys())
        accuracies = [model_scores[model]['test_accuracy'] for model in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color=sns.color_palette("husl", len(models)))
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison plot saved")
    
    def plot_confusion_matrix(self, evaluation_results: Dict[str, Any], save_dir: Path):
        """Plot confusion matrix for best model."""
        best_model = evaluation_results['best_model']
        conf_matrix = evaluation_results['model_scores'][best_model]['confusion_matrix']
        label_encoder = evaluation_results['label_encoder']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        
        plt.title(f'Confusion Matrix - {best_model}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Confusion matrix plot saved")
    
    def plot_feature_importance(self, evaluation_results: Dict[str, Any], save_dir: Path):
        """Plot feature importance if available."""
        try:
            best_model = evaluation_results['best_model']
            model_data = evaluation_results['model_scores'][best_model]
            
            # This is a placeholder - would need actual feature importance data
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Analysis - {best_model}', fontsize=16, fontweight='bold')
            plt.text(0.5, 0.5, 'Feature importance analysis\nwould be displayed here', 
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature importance plot saved")
            
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {e}")
    
    def create_all_plots(self, evaluation_results: Dict[str, Any], results_dir: Path):
        """Create all visualization plots."""
        print("Creating visualizations...")
        
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create different plots
        self.plot_model_comparison(evaluation_results, plots_dir)
        self.plot_confusion_matrix(evaluation_results, plots_dir)
        self.plot_feature_importance(evaluation_results, plots_dir)
        
        print(f"All plots saved to: {plots_dir}")
        logger.info("All visualization plots created successfully")