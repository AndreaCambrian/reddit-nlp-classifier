"""
Reddit NLP Classification Pipeline - Main Entry Point
Author: Andrea Oquendo Araujo
Course: Natural Language Processing AIE 1007
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            log_dir / f"reddit_nlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

from src.core.pipeline_manager import PipelineManager
from src.utils.project_info import print_project_banner, print_project_summary

def main():
    """Main execution function."""
    
    print_project_banner()
    
    try:
        # Initialize pipeline manager
        pipeline_manager = PipelineManager()
        
        # Run complete pipeline
        results = pipeline_manager.run_complete_pipeline()
        
        # Print summary
        print_project_summary(results)
        
        print("\nReddit NLP Classification Pipeline completed successfully!")
        print("Check the 'results' and 'logs' directories for detailed outputs.")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()
