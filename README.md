# Live Reddit NLP Classifier
**Andrea Oquendo Araujo - AIE1007**  
**Natural Language Processing Final Project**

## Academic Achievement
This project demonstrates advanced NLP capabilities with **88% accuracy** on live Reddit data classification, combining academic rigor with industry-ready deployment practices.

## Quick Start

### For Academic Demonstration (Local)
```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive demo
python interactive_demo.py
```

### For Production Deployment (Docker)
```bash
# Build and run with Docker (requires Docker installation)
docker build -t reddit-classifier .
docker run -it reddit-classifier
```

## Technical Excellence
- ✅ **Real-time Reddit API integration**
- ✅ **Advanced feature engineering** (TF-IDF + domain-specific features)
- ✅ **Balanced class classification** with Random Forest
- ✅ **Production-ready architecture** with Docker containerization
- ✅ **Interactive demonstration interface**

## Performance Results
| Metric | Score |
|--------|--------|
| Training Accuracy | 90.7% |
| Validation Accuracy | 78.0% |
| **Live Reddit Testing** | **88.0%** |

## Categories Classified
1. **Technology** - 60% accuracy on live posts
2. **Science** - 100% accuracy on live posts  
3. **Gaming** - 80% accuracy on live posts
4. **Politics** - 100% accuracy on live posts
5. **Fitness** - 100% accuracy on live posts

## Architecture
```
├── live_reddit_improved_classifier.py  # Core NLP system
├── interactive_demo.py                 # Professional demo interface
├── requirements.txt                    # Dependency management
├── Dockerfile                         # Container configuration
└── src/                              # Modular NLP components
```