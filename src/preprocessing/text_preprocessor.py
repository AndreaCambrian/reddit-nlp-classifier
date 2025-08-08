"""
Text Preprocessing Module for Reddit NLP Pipeline
"""

import re
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles all text preprocessing operations."""
    
    def __init__(self):
        """Initialize preprocessor with required NLTK components."""
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        logger.info("Text preprocessor initialized")
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                logger.info(f"Downloading NLTK data: {data}")
                nltk.download(data, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize text and apply lemmatization."""
        tokens = word_tokenize(text)
        
        # Remove punctuation and stopwords
        tokens = [token for token in tokens 
                 if token not in string.punctuation and token not in self.stop_words]
        
        # Lemmatize tokens
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return lemmatized
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataframe."""
        df_copy = df.copy()
        
        print("Cleaning text...")
        df_copy['cleaned_text'] = df_copy['text'].apply(self.clean_text)
        
        print("Advanced preprocessing...")
        df_copy['processed_tokens'] = df_copy['cleaned_text'].apply(self.tokenize_and_lemmatize)
        df_copy['processed_text'] = df_copy['processed_tokens'].apply(' '.join)
        
        logger.info(f"Preprocessing completed for {len(df_copy)} samples")
        
        return df_copy