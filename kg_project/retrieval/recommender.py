import torch
import torch.nn.functional as F
from typing import List, Tuple

class KGRecommender:
    """
    Implements similarity-based Information Retrieval for Knowledge Graph entities.
    """
    def __init__(self, embeddings: torch.Tensor, entity_to_id: dict):
        """
        Initialize with entity embeddings.
        
        Args:
            embeddings: Tensor of shape (num_entities, embedding_dim)
            entity_to_id: Dictionary mapping entity names to IDs.
        """
        # If embeddings are complex (e.g. from ComplEx model), convert to real representation
        # by concatenating real and imaginary parts.
        if torch.is_complex(embeddings):
            self.embeddings = torch.cat([embeddings.real, embeddings.imag], dim=-1)
        else:
            self.embeddings = embeddings
            
        self.entity_to_id = entity_to_id
        self.id_to_entity = {v: k for k, v in entity_to_id.items()}

    def find_similar(self, entity_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find top-k similar entities based on cosine similarity.
        
        Args:
            entity_name: The name of the query entity.
            top_k: Number of similar entities to return.
        """
        if entity_name not in self.entity_to_id:
            raise ValueError(f"Entity '{entity_name}' not found in mapping.")
        
        entity_id = self.entity_to_id[entity_name]
        query_vec = self.embeddings[entity_id].unsqueeze(0)
        
        # Normalize embeddings for cosine similarity
        norm_embeddings = F.normalize(self.embeddings, p=2, dim=1)
        norm_query = F.normalize(query_vec, p=2, dim=1)
        
        # Calculate similarities
        similarities = torch.mm(norm_embeddings, norm_query.t()).squeeze()
        
        # Get top-k indices (excluding the entity itself)
        vals, indices = torch.topk(similarities, k=top_k + 1)
        
        results = []
        for val, idx in zip(vals.tolist(), indices.tolist()):
            similar_entity = self.id_to_entity[idx]
            if similar_entity != entity_name:
                results.append((similar_entity, val))
        
        return results[:top_k]
