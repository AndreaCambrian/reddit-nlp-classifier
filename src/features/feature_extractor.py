"""
Feature Extraction Module for Reddit NLP Pipeline
"""

import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import Dict, Tuple, Any
import scipy.sparse

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Handles all feature extraction operations."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        logger.info("Feature extractor initialized")
    
    def extract_tfidf_features(self, texts: pd.Series) -> Tuple[np.ndarray, Any]:
        """Extract TF-IDF features."""
        print("Extracting TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF features: {tfidf_features.shape[1]} dimensions")
        
        return tfidf_features.toarray(), self.tfidf_vectorizer
    
    def extract_count_features(self, texts: pd.Series) -> Tuple[np.ndarray, Any]:
        """Extract count-based features."""
        print("Extracting count features...")
        
        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        count_features = self.count_vectorizer.fit_transform(texts)
        logger.info(f"Count features: {count_features.shape[1]} dimensions")
        
        return count_features.toarray(), self.count_vectorizer
    
    def extract_topic_features(self, texts: pd.Series, n_topics: int = 10) -> Tuple[np.ndarray, Any]:
        """Extract topic modeling features using LDA."""
        print("Extracting topic features...")
        
        # Use count vectorizer for LDA
        if self.count_vectorizer is None:
            count_features, _ = self.extract_count_features(texts)
        else:
            count_features = self.count_vectorizer.transform(texts).toarray()
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        
        topic_features = self.lda_model.fit_transform(count_features)
        logger.info(f"Topic features: {n_topics} topics extracted")
        
        return topic_features, self.lda_model
    
    def extract_statistical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract statistical text features."""
        print("Extracting statistical features...")
        
        features = []
        
        for _, row in df.iterrows():
            text = str(row['text'])
            
            stats = [
                len(text),  # Character count
                len(text.split()),  # Word count
                len([w for w in text.split() if len(w) > 6]),  # Long words
                text.count('!'),  # Exclamation marks
                text.count('?'),  # Question marks
                text.count('.'),  # Periods
                len(set(text.split()))  # Unique words
            ]
            
            features.append(stats)
        
        statistical_features = np.array(features)
        logger.info(f"Statistical features: {statistical_features.shape[1]} dimensions")
        
        return statistical_features
    
    def extract_all_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract all types of features."""
        print("Extracting text features...")
        
        # Use processed text for feature extraction
        texts = df['processed_text'] if 'processed_text' in df.columns else df['text']
        
        features = {}
        
        # Extract different types of features
        tfidf_features, tfidf_vectorizer = self.extract_tfidf_features(texts)
        count_features, count_vectorizer = self.extract_count_features(texts)
        topic_features, lda_model = self.extract_topic_features(texts)
        statistical_features = self.extract_statistical_features(df)
        
        # Combine all features
        combined_features = np.hstack([
            tfidf_features,
            topic_features,
            statistical_features
        ])
        
        features = {
            'X': combined_features,
            'y': df['subreddit'].values,
            'feature_names': {
                'tfidf': tfidf_vectorizer.get_feature_names_out() if hasattr(tfidf_vectorizer, 'get_feature_names_out') else [],
                'topics': [f'topic_{i}' for i in range(topic_features.shape[1])],
                'statistical': ['char_count', 'word_count', 'long_words', 'exclamations', 'questions', 'periods', 'unique_words']
            },
            'vectorizers': {
                'tfidf': tfidf_vectorizer,
                'count': count_vectorizer,
                'lda': lda_model
            }
        }
        
        logger.info("All features extracted successfully")
        return features