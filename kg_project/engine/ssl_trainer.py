import torch
import torch.nn as nn
import torch.optim as optim
from kg_project.models.ssm_encoder import KGSSMEncoder

class SSLTrainer:
    """
    Implements Self-Supervised Learning (SSL) for Knowledge Graphs.
    Focuses on a Masked Entity Prediction pretext task using an SSM architecture.
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int = 50, device: str = None, 
                 num_layers: int = 2, temp: float = 0.1, num_negatives: int = 64):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = KGSSMEncoder(num_entities, num_relations, dim, num_layers=num_layers).to(self.device)
        self.num_entities = num_entities
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)
        self.temp = temp
        self.num_negatives = num_negatives
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, head_ids: torch.Tensor, rel_ids: torch.Tensor, tail_ids: torch.Tensor):
        """
        Contrastive SSL Step: Maximize similarity between (h, r) prediction and true t,
        while minimizing similarity with negative samples.
        """
        self.encoder.train()
        self.optimizer.zero_grad()
        
        batch_size = head_ids.shape[0]
        h, r, t = head_ids.to(self.device), rel_ids.to(self.device), tail_ids.to(self.device)
        
        # 1. Predict state from prefix [h, r]
        predicted_state = self.encoder.predict_tail(h, r) # (Batch, Dim)
        
        # 2. Get Positive Tail Embedding
        pos_emb = self.encoder.entity_embeddings(t) # (Batch, Dim)
        
        # 3. Sample Negatives
        # For simplicity, we sample random entities. 
        # More advanced: hard negative sampling.
        neg_ids = torch.randint(0, self.num_entities, (batch_size, self.num_negatives), device=self.device)
        neg_emb = self.encoder.entity_embeddings(neg_ids) # (Batch, Negs, Dim)
        
        # 4. Calculate Scores (InfoNCE Logic)
        # Positive score: cosine similarity (dot product on normalized or just dot product)
        # We use dot product scaled by temperature
        pos_score = torch.sum(predicted_state * pos_emb, dim=-1, keepdim=True) / self.temp # (Batch, 1)
        
        # Negative scores
        # bmm: (Batch, 1, Dim) x (Batch, Dim, Negs) -> (Batch, 1, Negs)
        neg_score = torch.bmm(predicted_state.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1) / self.temp # (Batch, Negs)
        
        # Concatenate: [Pos, Neg1, Neg2, ...] -> Labels are all 0
        logits = torch.cat([pos_score, neg_score], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def run_ssl_epoch(self, mapped_triples: torch.Tensor, batch_size: int = 32):
        """
        Run one epoch of SSL pre-training.
        """
        total_loss = 0
        num_triples = mapped_triples.shape[0]
        
        for i in range(0, num_triples, batch_size):
            batch = mapped_triples[i : i + batch_size]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            
            loss = self.train_step(h, r, t)
            total_loss += loss
            
        return total_loss / (max(1, num_triples // batch_size))
