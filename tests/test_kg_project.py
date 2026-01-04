import sys
import os
import warnings

# Suppress internal PyTorch/PyKeen serialization/environment warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import pytest
import torch

# Add the project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_project.data.loader import TripleDataLoader
from kg_project.models.encoders import KGEncoderFactory, get_entity_embeddings
from kg_project.retrieval.recommender import KGRecommender
from kg_project.utils.benchmark import KGProfiler

@pytest.fixture
def nations_loader():
    return TripleDataLoader("nations")

def test_dataloader_loading(nations_loader):
    dataset = nations_loader.load_data()
    assert dataset is not None
    assert nations_loader.get_triples("training") is not None

def test_model_creation(nations_loader):
    nations_loader.load_data()
    factory = nations_loader.dataset.training
    model = KGEncoderFactory.create_model("transe", factory, embedding_dim=10)
    assert model is not None
    assert hasattr(model, 'entity_representations')

def test_embedding_extraction(nations_loader):
    nations_loader.load_data()
    factory = nations_loader.dataset.training
    model = KGEncoderFactory.create_model("distmult", factory, embedding_dim=10)
    embeddings = get_entity_embeddings(model)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[1] == 10

def test_recommender(nations_loader):
    nations_loader.load_data()
    entity_to_id = nations_loader.get_entity_mapping()
    num_entities = len(entity_to_id)
    dummy_embeddings = torch.randn(num_entities, 10)
    
    recommender = KGRecommender(dummy_embeddings, entity_to_id)
    sample_entity = list(entity_to_id.keys())[0]
    similar = recommender.find_similar(sample_entity, top_k=3)
    
    assert len(similar) == 3
    assert isinstance(similar[0][0], str)
    assert isinstance(similar[0][1], float)

def test_profiler():
    def dummy_func(x):
        return x * 2
    
    res, elapsed = KGProfiler.measure_time(dummy_func, 10)
    assert res == 20
    assert elapsed >= 0

if __name__ == "__main__":
    # If run directly as a script, execute pytest on itself
    pytest.main([__file__])
