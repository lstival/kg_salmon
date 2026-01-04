from pykeen.pipeline import pipeline
from typing import Dict, Any
import torch

class KGTrainer:
    """
    Manages the training of Knowledge Graph models using PyKeen's pipeline.
    """
    def __init__(self, dataset_name: str = "nations"):
        self.dataset_name = dataset_name

    def train(self, model_name: str, epochs: int = 5, embedding_dim: int = 50) -> Dict[str, Any]:
        """
        Train a model using the PyKeen pipeline.
        """
        import warnings
        # Broadly silence the torch serialization warning which is triggered inside pykeen
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
        
        print(f"Starting training for {model_name} on {self.dataset_name}...")
        
        result = pipeline(
            dataset=self.dataset_name,
            model=model_name,
            training_kwargs=dict(num_epochs=epochs),
            model_kwargs=dict(embedding_dim=embedding_dim),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_seed=42  # Explicitly set seed to silence the random seed warning
        )
        
        return {
            "model": result.model,
            "results": result,
            "training_time": result.training_loop_duration if hasattr(result, 'training_loop_duration') else 0
        }
