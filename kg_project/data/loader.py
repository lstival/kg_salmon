import pykeen.datasets
from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory
import pandas as pd
import numpy as np
from typing import Optional, Dict

class TripleDataLoader:
    """
    A utility class to load triples from PyKeen datasets or CSV files.
    """
    def __init__(self, dataset_name: str = "nations"):
        """
        Initialize the loader with a specific dataset name or file path.
        
        Args:
            dataset_name: The name of the PyKeen dataset or path to a CSV file.
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.triples_factory = None

    def load_from_csv(self, file_path: str, separator: str = ",", header: int = 0):
        """
        Load triples from a CSV file (h, r, t format).
        """
        print(f"Loading data from CSV: {file_path}")
        df = pd.read_csv(file_path, sep=separator, header=header)
        
        # Ensure we have 3 columns
        if df.shape[1] < 3:
            raise ValueError("CSV must have at least 3 columns (head, relation, tail)")
            
        # Extract triples as numpy array
        triples = df.iloc[:, :3].values.astype(str)
        
        # Create a TriplesFactory
        self.triples_factory = TriplesFactory.from_labeled_triples(triples)
        return self.triples_factory

    def load_data(self):
        """
        Load the dataset from PyKeen.
        """
        if self.dataset_name.endswith(".csv"):
            return self.load_from_csv(self.dataset_name)
            
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = get_dataset(dataset=self.dataset_name)
        self.triples_factory = self.dataset.training
        return self.dataset

    def get_triples(self, split: str = "training"):
        """
        Get triples for a specific split.
        """
        if self.triples_factory is None:
            self.load_data()
            
        if self.dataset is not None:
            if split == "training":
                return self.dataset.training.mapped_triples
            elif split == "testing":
                return self.dataset.testing.mapped_triples
            elif split == "validation":
                return self.dataset.validation.mapped_triples
        else:
            # For CSV, we treat everything as training for now or split manually if needed
            return self.triples_factory.mapped_triples

    def get_entity_mapping(self):
        """Returns the entity to ID mapping."""
        if self.triples_factory is None: self.load_data()
        return self.triples_factory.entity_to_id

    def get_relation_mapping(self):
        """Returns the relation to ID mapping."""
        if self.triples_factory is None: self.load_data()
        return self.triples_factory.relation_to_id
