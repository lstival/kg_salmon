from pykeen.pipeline import pipeline
from typing import Dict, Any, Optional
import torch
from kg_project.models.encoders import KGEncoderFactory

class KGTrainer:
    """
    Manages the training of Knowledge Graph models using PyKeen's pipeline or manual training loops.
    """
    def __init__(self, dataset_name: str = "nations"):
        self.dataset_name = dataset_name

    def train(self, model_name: str, epochs: int = 5, embedding_dim: int = 50, use_sbert: bool = False) -> Dict[str, Any]:
        """
        Train a model.
        """
        import warnings
        # Broadly silence the torch serialization warning which is triggered inside pykeen
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
        
        print(f"Starting training for {model_name} on {self.dataset_name}...")

        if model_name == "transformer_baseline":
            # For the transformer baseline, we implement a minimal training loop
            from pykeen.triples import TriplesFactory
            from kg_project.data.loader import TripleDataLoader
            loader = TripleDataLoader(self.dataset_name)
            tf = loader.load_data()
            if not isinstance(tf, TriplesFactory): # if it's a dataset object
                tf = tf.training

            model = KGEncoderFactory.create_model(model_name, tf, embedding_dim=embedding_dim)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            
            # Simple training loop for the baseline
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            print(f"Executing manual training loop for {model_name}...")
            triples = tf.mapped_triples.to(device)
            batch_size = 1024
            
            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, triples.shape[0], batch_size):
                    batch = triples[i:i+batch_size]
                    optimizer.zero_grad()
                    scores = model.predict_t(batch[:, :2])
                    loss = criterion(scores, batch[:, 2])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  {model_name} Epoch {epoch+1}/{epochs} | Loss (↓): {epoch_loss/(triples.shape[0]/batch_size):.4f}")
                
            return {
                "model": model,
                "results": None,
                "training_time": 1.0 # Approximate
            }
        
        result = pipeline(
            dataset=self.dataset_name,
            model=model_name,
            training_kwargs=dict(num_epochs=epochs),
            model_kwargs=dict(embedding_dim=embedding_dim),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_seed=42
        )
        
        return {
            "model": result.model,
            "results": result,
            "training_time": result.training_loop_duration if hasattr(result, 'training_loop_duration') else 0
        }
