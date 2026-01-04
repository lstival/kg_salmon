import torch
import numpy as np
from typing import Dict, List, Any

class KGEvaluator:
    """
    Calculates standard Knowledge Graph evaluation metrics for Link Prediction.
    Metrics: Hits@5, Hits@10, MRR (Mean Reciprocal Rank), MR (Mean Rank).
    """
    @staticmethod
    def calculate_metrics(ranks: List[int]) -> Dict[str, float]:
        """
        Calculates metrics based on a list of ranks (1-indexed).
        """
        if not ranks:
            return {"MR": 0.0, "MRR": 0.0, "Hits@5": 0.0, "Hits@10": 0.0}
            
        ranks_arr = np.array(ranks)
        mr = np.mean(ranks_arr)
        mrr = np.mean(1.0 / ranks_arr)
        hits_at_5 = np.mean(ranks_arr <= 5)
        hits_at_10 = np.mean(ranks_arr <= 10)
        
        return {
            "MR": float(mr),
            "MRR": float(mrr),
            "Hits@5": float(hits_at_5),
            "Hits@10": float(hits_at_10)
        }

    @staticmethod
    def get_rank(scores: torch.Tensor, true_id: int, filter_indices: torch.Tensor = None) -> int:
        """
        Returns the FILTERED rank of true_id based on scores.
        Filtered rank avoids penalizing the model for ranking other true triples high.
        """
        true_score = scores[true_id].item()
        
        if filter_indices is not None and filter_indices.numel() > 0:
            # Shift scores of other true triples to be very low so they don't affect rank
            # We use a copy to avoid mutating the original scores if reused
            scores = scores.clone()
            scores[filter_indices] = -1e9
            # Restore true score
            scores[true_id] = true_score
            
        # Count how many scores are GREATER than the true score
        rank = (scores > true_score).sum().item() + 1
        return int(rank)

    @staticmethod
    def evaluate_model(
        model: Any, 
        test_triples: torch.Tensor, 
        all_triples: torch.Tensor = None,
        entity_embeddings: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Unified evaluation interface with Filtered Ranking.
        """
        if entity_embeddings is not None:
            device = entity_embeddings.device
        elif hasattr(model, 'device'):
            device = model.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Build filter lookup if all_triples provided
        filter_dict = {}
        if all_triples is not None:
            for i in range(all_triples.shape[0]):
                h_f, r_f, t_f = all_triples[i, 0].item(), all_triples[i, 1].item(), all_triples[i, 2].item()
                if (h_f, r_f) not in filter_dict:
                    filter_dict[(h_f, r_f)] = []
                filter_dict[(h_f, r_f)].append(t_f)

        ranks = []
        # Sample evaluation for very large datasets to keep it responsive on laptop
        max_eval = 500
        indices = np.random.choice(len(test_triples), min(len(test_triples), max_eval), replace=False)
        subset = test_triples[indices]

        for triple in subset:
            h, r, t = triple[0].item(), triple[1].item(), triple[2].item()
            
            # Predict scores for all possible tails
            if hasattr(model, 'predict_t'): # PyKeen style
                scores = model.predict_t(torch.tensor([[h, r]], device=device)).squeeze()
            elif hasattr(model, 'predict_tail'):
                # Custom SSL-SSM style: Use predict_tail to avoid identity leak
                with torch.no_grad():
                    h_t = torch.tensor([h], device=device)
                    r_t = torch.tensor([r], device=device)
                    # No t passed here!
                    query_vec = model.predict_tail(h_t, r_t) 
                    scores = torch.mm(query_vec, entity_embeddings.t()).squeeze()
            else:
                raise ValueError("Model format not recognized for evaluation.")
            
            # Get other true tails for filtering
            f_indices = None
            if (h, r) in filter_dict:
                f_indices = torch.tensor(filter_dict[(h, r)], device=device)
            
            rank = KGEvaluator.get_rank(scores, t, filter_indices=f_indices)
            ranks.append(rank)
            
        return KGEvaluator.calculate_metrics(ranks)
