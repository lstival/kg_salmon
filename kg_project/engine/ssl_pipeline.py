import os
import torch
import warnings
from typing import Dict, Any
from kg_project.engine.ssl_trainer import SSLTrainer
from kg_project.retrieval.recommender import KGRecommender
from kg_project.data.loader import TripleDataLoader

class SSLPipeline:
    """
    Unified pipeline for Self-Supervised Learning on Knowledge Graphs.
    Handles pre-training, persistence, and recommendation.
    """
    def __init__(self, dataset_name: str, dim: int = 50, checkpoint_dir: str = "checkpoints", 
                 device: str = None, num_layers: int = 2, temp: float = 0.1, num_negatives: int = 64):
        self.dataset_name = dataset_name
        self.dim = dim
        self.num_layers = num_layers
        self.temp = temp
        self.num_negatives = num_negatives
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load Data
        self.loader = TripleDataLoader(dataset_name)
        self.loader.load_data()
        self.entity_to_id = self.loader.get_entity_mapping()
        
        num_entities = len(self.entity_to_id)
        num_relations = len(self.loader.get_relation_mapping())
        
        # Initialize Trainer (and SSM Encoder inside it)
        self.trainer = SSLTrainer(
            num_entities, 
            num_relations, 
            dim, 
            device=self.device, 
            num_layers=self.num_layers,
            temp=self.temp,
            num_negatives=self.num_negatives
        )

    def run_training(self, epochs: int = 15):
        """
        Executes the SSL pre-training phase.
        """
        print(f"\n>>> Starting SSL Pipeline Training on {self.dataset_name}...")
        triples = self.loader.get_triples("training")
        
        for epoch in range(1, epochs + 1):
            loss = self.trainer.run_ssl_epoch(triples)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:02d}/{epochs} | Loss (↓): {loss:.6f}")
        
        # Save model
        save_path = os.path.join(self.checkpoint_dir, f"ssm_ssl_{self.dataset_name}.pt")
        self.trainer.encoder.save(save_path)
        print(f"Model saved to {save_path}")

    def get_recommendations(self, entity_name: str, top_k: int = 5):
        """
        Performs retrieval using the learned SSL-SSM embeddings.
        """
        embeddings = self.trainer.encoder.get_all_embeddings()
        recommender = KGRecommender(embeddings, self.entity_to_id)
        return recommender.find_similar(entity_name, top_k=top_k)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the SSL-SSM model on the testing set with filtered ranking.
        """
        from kg_project.engine.evaluator import KGEvaluator
        test_triples = self.loader.get_triples("testing")
        if test_triples is None or test_triples.numel() == 0:
            test_triples = self.loader.get_triples("training")[:max(100, len(self.loader.get_triples("training"))//10)]
            
        # Get all triples for filtered evaluation
        all_triples_list = []
        for split in ["training", "testing", "validation"]:
            t = self.loader.get_triples(split)
            if t is not None:
                all_triples_list.append(t)
        all_triples = torch.cat(all_triples_list, dim=0) if all_triples_list else None
        
        embeddings = self.trainer.encoder.get_all_embeddings()
        
        # Pass the encoder directly; KGEvaluator will detect predict_tail
        return KGEvaluator.evaluate_model(
            self.trainer.encoder, 
            test_triples, 
            all_triples=all_triples,
            entity_embeddings=embeddings
        )

    def compare_retrieval(self, query_entities: list, top_k: int = 5):
        """
        Formatted output for SSL-based recommendations.
        """
        print(f"\n--- SSL-SSM Recommendations ({self.dataset_name}) ---")
        for entity in query_entities:
            try:
                similar = self.get_recommendations(entity, top_k=top_k)
                sim_str = ", ".join([f"{name} ({score:.2f})" for name, score in similar])
                print(f"  - {entity}: {sim_str}")
            except Exception as e:
                print(f"  - {entity}: Error during retrieval ({e})")
