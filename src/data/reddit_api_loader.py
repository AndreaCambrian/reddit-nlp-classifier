"""
Reddit API Data Loader
Extends the existing pipeline to support live Reddit data
"""

import praw
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class RedditAPILoader:
    """Handles loading data directly from Reddit API."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize Reddit API loader."""
        
        self.reddit = None
        self.credentials_path = credentials_path
        self.setup_reddit_connection()
        logger.info("Reddit API loader initialized")
    
    def setup_reddit_connection(self):
        """Setup connection to Reddit API."""
        
        try:
            if self.credentials_path and Path(self.credentials_path).exists():
                # Load credentials from file
                import importlib.util
                spec = importlib.util.spec_from_file_location("creds", self.credentials_path)
                creds = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(creds)
                
                self.reddit = praw.Reddit(
                    client_id=creds.REDDIT_CONFIG['client_id'],
                    client_secret=creds.REDDIT_CONFIG['client_secret'],
                    user_agent=creds.REDDIT_CONFIG['user_agent']
                )
                
                logger.info("Reddit API connection established")
                
            else:
                logger.warning("No Reddit credentials found. Using demo mode.")
                
        except Exception as e:
            logger.error(f"Failed to setup Reddit connection: {e}")
            self.reddit = None
    
    def fetch_subreddit_posts(self, subreddit_name: str, limit: int = 100) -> List[Dict]:
        """Fetch posts from a specific subreddit."""
        
        if not self.reddit:
            return self._get_demo_posts(subreddit_name, limit)
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            for post in subreddit.hot(limit=limit):
                if post.stickied:
                    continue
                
                full_text = post.title
                if post.selftext:
                    full_text += " " + post.selftext
                
                posts.append({
                    'text': full_text,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'score': post.score,
                    'created_utc': post.created_utc,
                    'num_comments': post.num_comments
                })
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit_name}: {e}")
            return self._get_demo_posts(subreddit_name, limit)
    
    def _get_demo_posts(self, subreddit_name: str, limit: int) -> List[Dict]:
        """Generate demo posts when API is not available."""
        
        demo_data = {
            'technology': [
                "Breaking: New quantum computer achieves unprecedented performance",
                "AI developments in machine learning show promising results",
                "Latest smartphone technology features advanced camera systems",
                "Cloud computing infrastructure revolutionizes data processing",
                "Cybersecurity advances protect against emerging threats"
            ],
            'science': [
                "Researchers discover new properties of dark matter",
                "Climate science reveals unexpected atmospheric patterns", 
                "Medical breakthrough in gene therapy treatment",
                "Physics experiment confirms theoretical predictions",
                "Biological research uncovers cellular mechanisms"
            ],
            'gaming': [
                "New game release sets industry records for player engagement",
                "Gaming hardware advances enable realistic graphics rendering",
                "Esports tournament draws millions of viewers worldwide",
                "Game development tools improve creator accessibility",
                "Virtual reality gaming reaches new immersion levels"
            ],
            'politics': [
                "Policy changes affect economic development strategies",
                "Legislative session addresses infrastructure improvements", 
                "Government initiatives focus on renewable energy transition",
                "Public health policies adapt to current challenges",
                "Educational reforms aim to improve student outcomes"
            ],
            'fitness': [
                "New workout routine improves cardiovascular health significantly",
                "Nutrition research reveals optimal dietary strategies",
                "Exercise equipment innovations enhance training efficiency",
                "Sports medicine advances help prevent common injuries",
                "Fitness tracking technology provides detailed health metrics"
            ]
        }
        
        posts = []
        import random
        
        available_posts = demo_data.get(subreddit_name, demo_data['technology'])
        
        for i in range(min(limit, len(available_posts))):
            posts.append({
                'text': available_posts[i],
                'subreddit': subreddit_name,
                'title': available_posts[i],
                'score': random.randint(50, 1000),
                'created_utc': datetime.now().timestamp(),
                'num_comments': random.randint(5, 200)
            })
        
        return posts