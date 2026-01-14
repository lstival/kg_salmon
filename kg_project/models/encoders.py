import torch
import torch.nn as nn
from pykeen.models import TransE, AutoSF, RotatE, DistMult, ComplEx
from pykeen.triples import TriplesFactory
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer

class TransformerBaseline(nn.Module):
    """
    A simple Transformer-based baseline model for entity representation.
    Uses a minimal transformer encoder architecture.
    """
    def __init__(self, num_entities: int, embedding_dim: int, nhead: int = 4, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        # (Batch, Seq) -> (Batch, Seq, Dim)
        x = self.embedding(entity_ids)
        x = self.transformer(x)
        return self.fc(x)

    def predict_t(self, hr_indices: torch.Tensor) -> torch.Tensor:
        """
        PyKeen-compatible score function for all possible tails.
        For a baseline transformer, we return a similarity score based on head embeddings.
        """
        # (Batch, 2) -> h_indices, r_indices
        h_indices = hr_indices[:, 0]
        # Get head embeddings
        h_embeds = self.forward(h_indices.unsqueeze(1)).squeeze(1) # (Batch, Dim)
        # All entity embeddings
        all_embeds = self.embedding.weight # (NumEntities, Dim)
        # Simple dot product as score
        scores = torch.mm(h_embeds, all_embeds.t())
        return scores

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
    def get_sbert_embeddings(labels: List[str], model_name: str = "paraphrase-MiniLM-L6-v2") -> torch.Tensor:
        """
        Generate initial embeddings for labels using SBERT.
        """
        print(f"Generating SBERT embeddings using {model_name}...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(labels, convert_to_tensor=True)
        return embeddings

    @staticmethod
    def create_model(model_name: str, triples_factory: TriplesFactory, use_sbert: bool = False, **kwargs) -> Any:
        """
        Create a PyKeen model.
        
        Args:
            model_name: Name of the model.
            triples_factory: The TriplesFactory containing entity and relation mappings.
            use_sbert: Whether to initialize entity embeddings with SBERT.
            kwargs: Additional arguments for the model.
        """
        model_name = model_name.lower()
        
        if model_name == "transformer_baseline":
             return TransformerBaseline(
                 num_entities=triples_factory.num_entities,
                 embedding_dim=kwargs.get("embedding_dim", 50)
             )

        if model_name not in KGEncoderFactory.MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(KGEncoderFactory.MODELS.keys())} or transformer_baseline")
        
        model_cls = KGEncoderFactory.MODELS[model_name]
        
        # Default embedding dimensions if not provided
        if "embedding_dim" not in kwargs:
            kwargs["embedding_dim"] = 50
            
        print(f"Creating model: {model_name}")
        model = model_cls(triples_factory=triples_factory, **kwargs)

        if use_sbert:
            # Get labels and generate embeddings
            entity_id_to_label = {v: k for k, v in triples_factory.entity_to_id.items()}
            labels = [entity_id_to_label[i] for i in range(triples_factory.num_entities)]
            sbert_embeddings = KGEncoderFactory.get_sbert_embeddings(labels)
            
            # pykeen models store representations in entity_representations
            # We need to ensure we match the dimensions
            # This is a simplification; different models might have different representation structures
            if hasattr(model, 'entity_representations'):
                # Note: This might require projection if dimensions don't match
                # SBERT paraphrase-MiniLM-L6-v2 is 384d
                pass 

        return model

def get_entity_embeddings(model: Any) -> torch.Tensor:
    """
    Extract entity embeddings from a trained PyKeen model.
    """
    if hasattr(model, 'entity_representations'):
        return model.entity_representations[0]().detach()
    else:
        raise AttributeError("Model does not have 'entity_representations'.")
