import warnings
import os
import torch

# Workaround for "Entry Point Not Found" DLL errors in Windows/Conda
# by ensuring torch is loaded before any other dependencies that might 
# trigger torchvision imports (e.g., PyKeen or SBERT).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

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
