"""
Interactive Demo for Live Reddit Classifier
Showcases real-time classification capabilities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from live_reddit_improved_classifier import LiveRedditImprovedClassifier
import time

class InteractiveDemo:
    """Interactive demo for live Reddit classification."""
    
    def __init__(self):
        """Initialize demo."""
        
        self.classifier = None
        print("INTERACTIVE REDDIT CLASSIFIER DEMO")
        print("Andrea Oquendo Araujo - AIE1007")
        print("Natural Language Processing Final Project")
        print("=" * 60)
    
    def load_or_train_model(self):
        """Load existing model or train new one."""
        
        print("\n1. MODEL SETUP")
        print("-" * 30)
        
        self.classifier = LiveRedditImprovedClassifier()
        
        if not self.classifier.reddit:
            print("Reddit API not available - demo mode limited")
            return False
        
        choice = input("\nOptions:\n[1] Train new model with fresh Reddit data\n[2] Quick demo with sample texts\nChoice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nTraining new model with live Reddit data...")
            results = self.classifier.run_full_live_pipeline()
            return results is not None
        else:
            print("\nUsing quick demo mode...")
            # Train on minimal data for demo
            training_df = self.classifier.collect_diverse_reddit_data(posts_per_subreddit=10)
            if training_df is not None and len(training_df) > 20:
                self.classifier.train_improved_model(training_df)
                return True
            else:
                print("Could not collect enough data for demo")
                return False
    
    def classify_user_input(self):
        """Let user input text for classification."""
        
        print("\n2. INTERACTIVE TEXT CLASSIFICATION")
        print("-" * 40)
        print("Enter text to classify or 'quit' to exit:")
        
        while True:
            user_text = input("\n> ").strip()
            
            if user_text.lower() == 'quit':
                break
            
            if len(user_text) < 10:
                print("Please enter longer text (at least 10 characters)")
                continue
            
            result = self.classifier.classify_text(user_text)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue
            
            print(f"\nText: '{user_text}'")
            print(f"Predicted Category: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            print("\nAll Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for category, prob in sorted_probs:
                print(f"  {category}: {prob:.3f}")
    
    def live_reddit_demo(self):
        """Demonstrate live Reddit classification."""
        
        print("\n3. LIVE REDDIT CLASSIFICATION DEMO")
        print("-" * 45)
        
        if not self.classifier.reddit:
            print("Reddit API not available for live demo")
            return
        
        subreddits = ['technology', 'science', 'gaming', 'politics', 'fitness']
        
        choice = input(f"\nChoose subreddit to analyze: {subreddits}\nOr 'all' for all subreddits: ").strip().lower()
        
        if choice == 'all':
            target_subreddits = subreddits
        elif choice in subreddits:
            target_subreddits = [choice]
        else:
            print("Invalid choice, using 'science'")
            target_subreddits = ['science']
        
        for subreddit in target_subreddits:
            print(f"\nAnalyzing live posts from r/{subreddit}...")
            
            try:
                reddit_sub = self.classifier.reddit.subreddit(subreddit)
                
                for i, post in enumerate(reddit_sub.hot(limit=3), 1):
                    if post.stickied:
                        continue
                    
                    full_text = post.title
                    if post.selftext and len(post.selftext) > 20:
                        full_text += " " + post.selftext[:200]
                    
                    if len(full_text) < 30:
                        continue
                    
                    result = self.classifier.classify_text(full_text)
                    
                    if 'error' not in result:
                        predicted = result['predicted_class']
                        confidence = result['confidence']
                        correct = predicted.lower() == subreddit.lower()
                        
                        status = "CORRECT" if correct else "INCORRECT"
                        print(f"\nPost {i}: {post.title[:50]}...")
                        print(f"Actual: {subreddit} | Predicted: {predicted} | {status}")
                        print(f"Confidence: {confidence:.3f}")
                        
                        if not correct:
                            sorted_probs = sorted(result['all_probabilities'].items(), 
                                                key=lambda x: x[1], reverse=True)[:3]
                            print(f"Top 3: {sorted_probs}")
                    
                    time.sleep(0.2)
                    
            except Exception as e:
                print(f"Error analyzing r/{subreddit}: {e}")
    
    def sample_text_demo(self):
        """Demo with sample texts."""
        
        print("\n4. SAMPLE TEXT CLASSIFICATION DEMO")
        print("-" * 45)
        
        sample_texts = [
            ("New iPhone 15 features breakthrough AI chip technology", "technology"),
            ("Scientists discover new cancer treatment using immunotherapy", "science"),
            ("Call of Duty releases new multiplayer maps and weapons", "gaming"),
            ("Election results show surprising voter turnout patterns", "politics"),
            ("Best workout routine for building muscle and losing fat", "fitness"),
            ("Apple CEO announces major software update with AI features", "technology"),
            ("Research shows exercise improves mental health significantly", "science"),
            ("PlayStation 5 exclusive games coming this holiday season", "gaming"),
            ("Congress passes new healthcare legislation after debate", "politics"),
            ("Nutrition guide for bodybuilders and strength athletes", "fitness")
        ]
        
        correct_predictions = 0
        
        for text, expected_category in sample_texts:
            result = self.classifier.classify_text(text)
            
            if 'error' not in result:
                predicted = result['predicted_class']
                confidence = result['confidence']
                correct = predicted.lower() == expected_category.lower()
                
                if correct:
                    correct_predictions += 1
                
                status = "CORRECT" if correct else "INCORRECT"
                print(f"\nText: {text}")
                print(f"Expected: {expected_category} | Predicted: {predicted} | {status}")
                print(f"Confidence: {confidence:.3f}")
        
        accuracy = correct_predictions / len(sample_texts)
        print(f"\nSample Demo Accuracy: {accuracy:.1%} ({correct_predictions}/{len(sample_texts)})")
    
    def run_complete_demo(self):
        """Run the complete interactive demo."""
        
        # Step 1: Setup model
        if not self.load_or_train_model():
            print("Could not setup model. Exiting demo.")
            return
        
        print(f"\nModel ready! Categories: {list(self.classifier.label_encoder.classes_)}")
        
        while True:
            print("\n" + "="*60)
            print("DEMO OPTIONS:")
            print("[1] Classify your own text")
            print("[2] Live Reddit analysis")
            print("[3] Sample text demonstration")
            print("[4] Exit demo")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                self.classify_user_input()
            elif choice == "2":
                self.live_reddit_demo()
            elif choice == "3":
                self.sample_text_demo()
            elif choice == "4":
                print("\nDemo complete! Thank you for testing the classifier.")
                break
            else:
                print("Invalid choice. Please select 1-4.")

def main():
    """Main demo function."""
    
    demo = InteractiveDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()