import torch
from pykeen.models import TransE, AutoSF, RotatE, DistMult, ComplEx
from pykeen.triples import TriplesFactory
from typing import Dict, Any

class KGEncoderFactory:
    """
    Factory class to create and manage Knowledge Graph Encoders.
    """
    MODELS = {
        "transe": TransE,
        "autosf": AutoSF,
        "rotate": RotatE,
        "distmult": DistMult,
        "complex": ComplEx
    }

    @staticmethod
    def create_model(model_name: str, triples_factory: TriplesFactory, **kwargs) -> Any:
        """
        Create a PyKeen model.
        
        Args:
            model_name: Name of the model (transe, autosf, rotate, distmult, cooccurrence).
            triples_factory: The TriplesFactory containing entity and relation mappings.
            kwargs: Additional arguments for the model.
        """
        model_name = model_name.lower()
        if model_name not in KGEncoderFactory.MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(KGEncoderFactory.MODELS.keys())}")
        
        model_cls = KGEncoderFactory.MODELS[model_name]
        
        # Default embedding dimensions if not provided
        if "embedding_dim" not in kwargs:
            kwargs["embedding_dim"] = 50
            
        print(f"Creating model: {model_name}")
        return model_cls(triples_factory=triples_factory, **kwargs)

def get_entity_embeddings(model: Any) -> torch.Tensor:
    """
    Extract entity embeddings from a trained PyKeen model.
    """
    if hasattr(model, 'entity_representations'):
        return model.entity_representations[0]().detach()
    else:
        raise AttributeError("Model does not have 'entity_representations'.")
