import pykeen.datasets
from pykeen.datasets import get_dataset
from typing import Optional

class TripleDataLoader:
    """
    A utility class to load triples from PyKeen datasets.
    """
    def __init__(self, dataset_name: str = "nations"):
        """
        Initialize the loader with a specific dataset.
        
        Args:
            dataset_name: The name of the PyKeen dataset to load.
        """
        self.dataset_name = dataset_name
        self.dataset = None

    def load_data(self):
        """
        Load the dataset from PyKeen.
        """
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = get_dataset(dataset=self.dataset_name)
        return self.dataset

    def get_triples(self, split: str = "training"):
        """
        Get triples for a specific split.
        
        Args:
            split: One of 'training', 'testing', or 'validation'.
        """
        if self.dataset is None:
            self.load_data()
        
        if split == "training":
            return self.dataset.training.mapped_triples
        elif split == "testing":
            return self.dataset.testing.mapped_triples
        elif split == "validation":
            return self.dataset.validation.mapped_triples
        else:
            raise ValueError(f"Invalid split: {split}")

    def get_entity_mapping(self):
        """Returns the entity to ID mapping."""
        return self.dataset.entity_to_id

    def get_relation_mapping(self):
        """Returns the relation to ID mapping."""
        return self.dataset.relation_to_id
