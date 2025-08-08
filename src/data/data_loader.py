"""
Data Loading Module for Reddit NLP Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and initial processing of Reddit data."""
    
    def __init__(self):
        """Initialize data loader."""
        self.data_path = None
        logger.info("Data loader initialized")
    
    def load_reddit_data(self) -> pd.DataFrame:
        """Load Reddit data from CSV file."""
        
        # Try to find the data file
        possible_paths = [
            'reddit_data_processed.csv',
            'data/reddit_data_processed.csv',
            '../reddit_data_processed.csv'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                self.data_path = path
                break
        
        if not self.data_path:
            raise FileNotFoundError("Reddit data file not found.")
        
        # Load the data
        df = pd.read_csv(self.data_path)
        
        # Basic validation
        required_columns = ['text', 'subreddit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean any missing values
        df = df.dropna(subset=['text', 'subreddit'])
        
        # Log statistics
        logger.info(f"Loaded data from: {Path(self.data_path).name}")
        logger.info(f"Data loaded successfully: {len(df)} samples")
        
        return df
    
    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """Get basic statistics about the loaded data."""
        
        stats = {
            'total_samples': len(df),
            'unique_subreddits': df['subreddit'].nunique(),
            'class_distribution': df['subreddit'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'min_text_length': df['text'].str.len().min(),
            'max_text_length': df['text'].str.len().max()
        }
        
        return stats
