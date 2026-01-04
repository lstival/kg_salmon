import torch
import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kg_project.data.loader import TripleDataLoader
from kg_project.engine.ssl_trainer import SSLTrainer
from kg_project.models.ssm_encoder import KGSSMEncoder

def test_dataloader_consistency():
    """
    Check if the DataLoader consistently maps the same names to the same IDs.
    """
    dataset_name = "nations"
    loader = TripleDataLoader(dataset_name)
    loader.load_data()
    
    mapping1 = loader.get_entity_mapping()
    
    # Reload and check
    loader2 = TripleDataLoader(dataset_name)
    loader2.load_data()
    mapping2 = loader2.get_entity_mapping()
    
    assert mapping1 == mapping2
    assert len(mapping1) > 0

def test_training_stability():
    """
    Verify that SSL loss decreases or stays stable during training.
    """
    dataset_name = "nations"
    loader = TripleDataLoader(dataset_name)
    loader.load_data()
    triples = loader.get_triples("training")
    
    num_entities = len(loader.get_entity_mapping())
    num_relations = len(loader.get_relation_mapping())
    
    trainer = SSLTrainer(num_entities, num_relations, dim=16)
    
    # Initial loss
    loss_start = trainer.run_ssl_epoch(triples[:32])
    
    # Train for a few epochs
    for _ in range(5):
        trainer.run_ssl_epoch(triples[:32])
        
    loss_end = trainer.run_ssl_epoch(triples[:32])
    
    # Loss should generally decrease for a simple subset
    assert loss_end <= loss_start or np.isclose(loss_end, loss_start, atol=1e-2)

def test_ssm_reproducibility():
    """
    Ensure the same inputs to SSM results in same embeddings.
    """
    dim = 16
    encoder = KGSSMEncoder(num_entities=10, num_relations=5, dim=dim)
    encoder.eval()
    
    h, r, t = torch.tensor([1]), torch.tensor([2]), torch.tensor([3])
    
    with torch.no_grad():
        out1 = encoder.encode_path(h, r, t)
        out2 = encoder.encode_path(h, r, t)
        
    assert torch.allclose(out1, out2)

if __name__ == "__main__":
    pytest.main([__file__])
