"""
Configuration Module for Reddit NLP Pipeline
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the pipeline."""
    
    def __init__(self):
        """Initialize configuration."""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_file = "reddit_data_processed.csv"
        self.results_dir = self.project_root / "results"
        self.logs_dir = self.project_root / "logs"
        
        # Model parameters
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        
        # Feature extraction parameters
        self.max_features = 1000
        self.ngram_range = (1, 2)
        self.min_df = 2
        self.max_df = 0.95
        self.n_topics = 10
        
        logger.info("Configuration initialized")
    
    def get_data_path(self):
        """Get the path to the data file."""
        return self.project_root / self.data_file
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)