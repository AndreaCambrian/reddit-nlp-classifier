"""
Live Reddit Improved Classifier
Collects fresh data from Reddit and trains improved model
"""

import praw
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from reddit_credentials import REDDIT_CONFIG

class LiveRedditImprovedClassifier:
    """Collect live Reddit data and train improved classifier."""
    
    def __init__(self):
        """Initialize live collector and classifier."""
        
        self.reddit = None
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        
        print("LIVE REDDIT IMPROVED CLASSIFIER")
        print("Andrea Oquendo Araujo - AIE1007")
        print("=" * 50)
        
        self.setup_reddit_api()
    
    def setup_reddit_api(self):
        """Set up Reddit API connection."""
        
        try:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CONFIG['client_id'],
                client_secret=REDDIT_CONFIG['client_secret'],
                user_agent=REDDIT_CONFIG['user_agent']
            )
            
            # Test connection
            self.reddit.auth.scopes()
            print("Reddit API connected successfully!")
            return True
            
        except Exception as e:
            print(f"Reddit API connection failed: {e}")
            return False
    
    def collect_diverse_reddit_data(self, posts_per_subreddit=60):
        """Collect diverse, fresh data from multiple subreddits."""
        
        if not self.reddit:
            print("No Reddit connection")
            return None
        
        print(f"Collecting {posts_per_subreddit} posts from each subreddit...")
        
        # Improved subreddit selection with more options
        subreddit_categories = {
            'technology': ['technology', 'gadgets', 'tech', 'apple', 'android', 'hardware'],
            'science': ['science', 'askscience', 'EverythingScience', 'biology', 'chemistry'],
            'gaming': ['gaming', 'Games', 'pcgaming', 'nintendo', 'PS5', 'xbox'],
            'politics': ['politics', 'PoliticalDiscussion', 'worldnews', 'news'],
            'fitness': ['fitness', 'bodybuilding', 'loseit', 'workout', 'nutrition']
        }
        
        all_posts = []
        
        for category, subreddit_list in subreddit_categories.items():
            category_posts = []
            
            print(f"\nCollecting {category} posts...")
            
            for subreddit_name in subreddit_list:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Try both hot and new posts for more variety
                    post_sources = [subreddit.hot(limit=50), subreddit.new(limit=30)]
                    
                    for post_source in post_sources:
                        for post in post_source:
                            # Be more flexible with post types
                            if post.stickied:
                                continue
                            
                            # Combine title and text
                            full_text = post.title
                            if hasattr(post, 'selftext') and post.selftext and len(post.selftext) > 20:
                                full_text += " " + post.selftext[:500]
                            
                            # More lenient filtering
                            if len(full_text) > 20 and len(full_text) < 2000:
                                category_posts.append({
                                    'text': full_text,
                                    'subreddit': category,
                                    'original_subreddit': subreddit_name,
                                    'score': getattr(post, 'score', 0),
                                    'title': post.title
                                })
                            
                            time.sleep(0.05)  # Faster collection
                            
                            if len(category_posts) >= posts_per_subreddit:
                                break
                        
                        if len(category_posts) >= posts_per_subreddit:
                            break
                    
                    if len(category_posts) >= posts_per_subreddit:
                        break
                        
                except Exception as e:
                    print(f"  Error with r/{subreddit_name}: {e}")
                    continue
            
            print(f"  Collected {len(category_posts)} {category} posts")
            all_posts.extend(category_posts[:posts_per_subreddit])
        
        df = pd.DataFrame(all_posts)
        
        if len(df) > 0:
            print(f"\nTotal collected: {len(df)} posts")
            print("Distribution:")
            print(df['subreddit'].value_counts())
            
            # Save fresh data
            df.to_csv('live_reddit_data.csv', index=False)
            print("Saved to live_reddit_data.csv")
        
        return df
    
    def create_advanced_features(self, texts):
        """Create advanced domain-specific features."""
        
        features = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Technology keywords (more specific)
            tech_keywords = [
                'iphone', 'android', 'apple', 'google', 'microsoft', 'ai', 'artificial intelligence',
                'software', 'app', 'algorithm', 'coding', 'programming', 'tech', 'digital',
                'computer', 'laptop', 'smartphone', 'device', 'bitcoin', 'crypto', 'blockchain'
            ]
            tech_score = sum(1 for word in tech_keywords if word in text_lower)
            
            # Science keywords (more specific)
            science_keywords = [
                'research', 'study', 'experiment', 'scientists', 'discovery', 'data', 'analysis',
                'hypothesis', 'theory', 'evidence', 'medical', 'health', 'cancer', 'covid',
                'vaccine', 'climate', 'space', 'nasa', 'physics', 'chemistry', 'biology'
            ]
            science_score = sum(1 for word in science_keywords if word in text_lower)
            
            # Gaming keywords (more specific)
            gaming_keywords = [
                'game', 'gaming', 'gamer', 'play', 'xbox', 'playstation', 'nintendo', 'steam',
                'esports', 'tournament', 'fps', 'rpg', 'mmo', 'console', 'pc gaming',
                'graphics', 'gameplay', 'multiplayer', 'single player'
            ]
            gaming_score = sum(1 for word in gaming_keywords if word in text_lower)
            
            # Politics keywords (more specific)
            politics_keywords = [
                'election', 'vote', 'voting', 'government', 'congress', 'senate', 'president',
                'political', 'politics', 'democrat', 'republican', 'policy', 'law', 'legislation',
                'trump', 'biden', 'campaign', 'candidate'
            ]
            politics_score = sum(1 for word in politics_keywords if word in text_lower)
            
            # Fitness keywords (more specific)
            fitness_keywords = [
                'workout', 'exercise', 'fitness', 'gym', 'training', 'muscle', 'protein',
                'weight', 'lifting', 'cardio', 'running', 'diet', 'nutrition', 'calories',
                'bodybuilding', 'strength', 'CrossFit', 'yoga'
            ]
            fitness_score = sum(1 for word in fitness_keywords if word in text_lower)
            
            # Advanced text features
            words = text.split()
            sentences = text.split('.')
            
            feature_vector = [
                tech_score / len(words) if words else 0,      # Normalized scores
                science_score / len(words) if words else 0,
                gaming_score / len(words) if words else 0,
                politics_score / len(words) if words else 0,
                fitness_score / len(words) if words else 0,
                len(words),                                    # Word count
                len(text),                                     # Character count
                len(sentences),                                # Sentence count
                text.count('?'),                              # Question marks
                text.count('!'),                              # Exclamation marks
                len([w for w in words if len(w) > 6]) / len(words) if words else 0,  # Long words ratio
                len(set(words)) / len(words) if words else 0,   # Unique words ratio
                text.count('http'),           # URLs
                text.count('@'),              # Mentions  
                len(re.findall(r'[A-Z]+', text)),  # Caps
                text.count('$'),              # Money symbols
                len(re.findall(r'\d+', text)) # Numbers
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_improved_model(self, df):
        """Train improved model on live Reddit data."""
        
        print("Training improved model on live Reddit data...")
        
        texts = df['text'].tolist()
        labels = df['subreddit'].tolist()
        
        # Advanced TF-IDF with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.7,
            ngram_range=(1, 3),  # Include trigrams
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        
        # Domain-specific features
        domain_features = self.create_advanced_features(texts)
        
        # Combine features
        X = np.hstack([tfidf_features, domain_features])
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Compute balanced weights
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(y), 
                                           y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Train ensemble model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,  # Use computed weights
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Improved model trained!")
        print(f"Training accuracy: {self.model.score(X_train, y_train):.3f}")
        print(f"Validation accuracy: {accuracy:.3f}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Detailed classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(f"\nDetailed Classification Report:")
        print(report)
        
        return accuracy
    
    def classify_text(self, text):
        """Classify text with improved model."""
        
        if not self.model:
            return {"error": "Model not trained"}
        
        try:
            # TF-IDF features
            tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()
            
            # Domain features
            domain_features = self.create_advanced_features([text])
            
            # Combine
            X = np.hstack([tfidf_features, domain_features])
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities.max()
            
            # All probabilities
            prob_dict = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[class_name] = float(probabilities[i])
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': prob_dict
            }
            
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
    
    def test_on_new_live_data(self):
        """Test improved model on completely new live Reddit data."""
        
        print("\nTESTING ON NEW LIVE REDDIT DATA")
        print("=" * 50)
        
        subreddits = ['technology', 'science', 'gaming', 'politics', 'fitness']
        all_results = {}
        
        for subreddit in subreddits:
            print(f"\nTesting on fresh r/{subreddit} posts...")
            
            try:
                reddit_sub = self.reddit.subreddit(subreddit)
                test_posts = []
                
                for post in reddit_sub.hot(limit=10):
                    if post.stickied:
                        continue
                    
                    full_text = post.title
                    if post.selftext and len(post.selftext) > 20:
                        full_text += " " + post.selftext[:300]
                    
                    if len(full_text) > 30:
                        test_posts.append({
                            'text': full_text,
                            'title': post.title,
                            'subreddit': subreddit
                        })
                        
                        if len(test_posts) >= 5:  # Test on 5 posts per subreddit
                            break
                    
                    time.sleep(0.1)
                
                if not test_posts:
                    print(f"  No valid posts found in r/{subreddit}")
                    continue
                
                correct = 0
                total = len(test_posts)
                
                for i, post in enumerate(test_posts, 1):
                    result = self.classify_text(post['text'])
                    
                    if 'error' not in result:
                        predicted = result['predicted_class']
                        confidence = result['confidence']
                        correct_prediction = predicted.lower() == subreddit.lower()
                        
                        if correct_prediction:
                            correct += 1
                        
                        status = "CORRECT" if correct_prediction else "INCORRECT"
                        print(f"  {status} Post {i}: {post['title'][:60]}...")
                        print(f"      Predicted: {predicted} ({confidence:.3f})")
                        
                        # Show top 3 predictions
                        sorted_probs = sorted(result['all_probabilities'].items(), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        print(f"      Top 3: {sorted_probs}")
                
                accuracy = correct / total if total > 0 else 0
                print(f"  r/{subreddit} Accuracy: {accuracy:.1%} ({correct}/{total})")
                
                all_results[subreddit] = {'correct': correct, 'total': total}
                
            except Exception as e:
                print(f"  Error testing r/{subreddit}: {e}")
        
        # Overall results
        total_correct = sum(r['correct'] for r in all_results.values())
        total_posts = sum(r['total'] for r in all_results.values())
        overall_accuracy = total_correct / total_posts if total_posts > 0 else 0
        
        print(f"\nFINAL IMPROVED RESULTS:")
        print(f"Total posts tested: {total_posts}")
        print(f"Correctly classified: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.1%}")
        
        if overall_accuracy > 0.6:
            print("EXCELLENT! Significant improvement achieved!")
        elif overall_accuracy > 0.4:
            print("GOOD! Solid improvement over previous version!")
        else:
            print("Some improvement, but more work needed!")
        
        return all_results
    
    def run_full_live_pipeline(self):
        """Run complete live Reddit pipeline."""
        
        # Step 1: Collect fresh training data
        training_df = self.collect_diverse_reddit_data(posts_per_subreddit=40)
        
        if training_df is None or len(training_df) < 50:
            print("Insufficient training data collected")
            return
        
        # Step 2: Train improved model
        accuracy = self.train_improved_model(training_df)
        
        if accuracy < 0.5:
            print("Model performance may be limited")
        
        # Step 3: Test on completely new live data
        results = self.test_on_new_live_data()
        
        return results

def main():
    """Main function."""
    
    classifier = LiveRedditImprovedClassifier()
    
    if classifier.reddit:
        print("Starting live Reddit classification pipeline...")
        results = classifier.run_full_live_pipeline()
        
        if results:
            print("\nLIVE REDDIT CLASSIFIER PIPELINE COMPLETE!")
            print("Your improved model is ready for presentation!")
    else:
        print("Cannot proceed without Reddit API connection")

if __name__ == "__main__":
    main()