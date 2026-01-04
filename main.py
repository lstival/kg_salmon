import warnings
import os

# Note: We avoid importing torchvision or filtering it here to prevent 
# "Entry Point Not Found" DLL errors in mismatched environments.

from kg_project.config import DEFAULT_CONFIG
from kg_project.engine.runner import ExperimentRunner

def main():
    print(">>> Initializing Knowledge Graph Experiment Pipeline...")
    
    # Use the centralized DEFAULT_CONFIG from kg_project/config.py
    config = DEFAULT_CONFIG
    
    runner = ExperimentRunner(config)
    runner.run_all()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
