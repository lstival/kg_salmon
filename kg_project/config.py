from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class KGConfig:
    """
    Centralized configuration for Knowledge Graph experiments.
    """
    # Dataset selection (e.g., 'nations', 'umls', 'kinships', 'fb15k237')
    dataset_name: str = "umls"
    
    # Model selection (list of models to benchmark)
    models: List[str] = field(default_factory=lambda: ["transe", "rotate", "distmult", "autosf", "complex", "transformer_baseline"])
    
    # NLP parameters
    use_sbert: bool = True
    sbert_model: str = "paraphrase-MiniLM-L6-v2"
    
    # Training parameters
    epochs: int = 25
    embedding_dim: int = 64
    num_layers: int = 3
    ssl_temp: float = 0.1 # Temperature for InfoNCE
    ssl_negatives: int = 64 # Negatives per batch sample
    
    # Retrieval parameters
    top_k: int = 5
    num_query_entities: int = 3
    
    # Visualization parameters
    use_tsne: bool = True
    tsne_perplexity: int = 30
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

# Default configuration (User requested: 'fb15k237')
DEFAULT_CONFIG = KGConfig(
    dataset_name="fb15k237",
    epochs=25,
    num_layers=3, 
    num_query_entities=5
)

# Challenging Benchmark Config (Standard for Research Papers)
FB15K_CONFIG = KGConfig(
    dataset_name="fb15k237",
    epochs=10, # Fewer epochs initially as training will take longer
    num_query_entities=5
)

# Middle-ground Config
CODEX_SMALL_CONFIG = KGConfig(
    dataset_name="codexsmall",
    epochs=15,
    num_query_entities=5
)
