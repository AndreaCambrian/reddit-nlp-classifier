"""
Project Information and Utilities Module
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def print_project_banner():
    """Print project banner with information."""
    print("=" * 80)
    print("REDDIT NLP CLASSIFICATION PIPELINE")
    print("=" * 80)
    print("Author: Andrea Oquendo Araujo")
    print("Course: Natural Language Processing AIE 1007")
    print("Project: Reddit Post Classification using Machine Learning")
    print("=" * 80)

def print_project_summary(results: Dict[str, Any]):
    """Print summary of pipeline results."""
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    if 'data_info' in results:
        data_info = results['data_info']
        print(f"Dataset: {data_info['total_samples']} Reddit posts")
        print(f"Classes: {len(data_info['classes'])} subreddits")
        
        print(f"\nCLASS DISTRIBUTION:")
        for class_name, count in data_info['class_distribution'].items():
            print(f"  {class_name}: {count} posts")
    
    if 'evaluation' in results:
        evaluation = results['evaluation']
        if 'best_model' in evaluation:
            print(f"\nBest Model: {evaluation['best_model']}")
            print(f"Best Accuracy: {evaluation['best_accuracy']:.4f}")
    
    if 'pipeline_time' in results:
        print(f"\nTotal Pipeline Time: {results['pipeline_time']:.2f} seconds")
    
    print("\nPipeline completed successfully!")
    print("Check 'results/' directory for detailed outputs and visualizations.")

def get_project_info():
    """Get project information dictionary."""
    return {
        'name': 'Reddit NLP Classification Pipeline',
        'author': 'Andrea Oquendo Araujo',
        'course': 'Natural Language Processing AIE 1007',
        'description': 'Machine Learning pipeline for classifying Reddit posts by subreddit',
        'features': [
            'Text preprocessing with NLTK',
            'TF-IDF feature extraction',
            'Topic modeling with LDA',
            'Multiple ML algorithms comparison',
            'Comprehensive evaluation metrics',
            'Automated visualization generation'
        ]
    }