import warnings
import torch
from typing import Dict, Any
from kg_project.config import KGConfig
from kg_project.data.loader import TripleDataLoader
from kg_project.engine.trainer import KGTrainer
from kg_project.retrieval.recommender import KGRecommender
from kg_project.utils.benchmark import KGProfiler
from kg_project.models.encoders import get_entity_embeddings
from kg_project.models.explanations import explain_models

from kg_project.engine.ssl_pipeline import SSLPipeline

from kg_project.engine.evaluator import KGEvaluator

class ExperimentRunner:
    """
    Orchestrates Knowledge Graph experiments based on a configuration.
    """
    def __init__(self, config: KGConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Suppress warnings for clean execution
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

    def run_all(self):
        """
        Execute the full pipeline: Baselines vs SSL.
        """
        print(f"=== Starting Experiment (Device: {self.device}) ===")
        print(f"Dataset: {self.config.dataset_name}")
        
        # 1. SSL Pipeline (Main Objective)
        ssl_pipe = SSLPipeline(
            self.config.dataset_name, 
            dim=self.config.embedding_dim,
            device=self.device,
            num_layers=self.config.num_layers,
            temp=self.config.ssl_temp,
            num_negatives=self.config.ssl_negatives
        )
        ssl_pipe.run_training(epochs=self.config.epochs)
        ssl_metrics = ssl_pipe.evaluate()
        
        # 2. Baseline Comparison
        loader = TripleDataLoader(self.config.dataset_name)
        loader.load_data()
        
        # Get all triples for filtered evaluation
        all_triples_list = []
        for split in ["training", "testing", "validation"]:
            t = loader.get_triples(split)
            if t is not None:
                all_triples_list.append(t)
        all_triples = torch.cat(all_triples_list, dim=0) if all_triples_list else None
        
        # Fix: Explicitly check for None/empty tensor to avoid ambiguous boolean evaluation
        test_triples = loader.get_triples("testing")
        if test_triples is None or test_triples.numel() == 0:
            test_triples = loader.get_triples("training")[:max(100, len(loader.get_triples("training"))//10)]
            
        entity_to_id = loader.get_entity_mapping()
        trainer = KGTrainer(self.config.dataset_name)
        
        results_summary = []
        sample_entities = list(entity_to_id.keys())[:self.config.num_query_entities]

        # Process standard PyKeen models as baselines
        for model_name in self.config.models:
            print(f"\n>>> Baseline: {model_name}")
            
            train_func = lambda: trainer.train(
                model_name, 
                epochs=self.config.epochs, 
                embedding_dim=self.config.embedding_dim
            )
            train_result, elapsed = KGProfiler.measure_time(train_func)
            
            model = train_result["model"]
            gflops = KGProfiler.estimate_gflops(model)
            
            # Link Prediction Evaluation (Filtered)
            metrics = KGEvaluator.evaluate_model(model, test_triples, all_triples=all_triples)
            
            results_summary.append({
                "model": model_name,
                "time": elapsed,
                "gflops": gflops,
                "metrics": metrics
            })
            
        # Final comparison output
        print("\n" + "="*50)
        ssl_pipe.compare_retrieval(sample_entities, top_k=self.config.top_k)
        
        # Add SSL metrics to the final summary table
        results_summary.insert(0, {
            "model": "SSL-SSM (Ours)",
            "time": 0.0, # Training time tracked separately
            "gflops": 0.0,
            "metrics": ssl_metrics
        })
        
        self._print_summary(results_summary)

    def _print_summary(self, summary):
        print("\n=== Comprehensive Journal-Grade Summary ===")
        header = f"{'Model':<15} | {'Hits@5':<8} | {'Hits@10':<8} | {'MRR':<8} | {'MR':<8}"
        print(header)
        print("-" * len(header))
        for res in summary:
            m = res['metrics']
            print(f"{res['model']:<15} | {m['Hits@5']:<8.4f} | {m['Hits@10']:<8.4f} | {m['MRR']:<8.4f} | {m['MR']:<8.1f}")
