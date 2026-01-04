import torch
import torch.nn as nn
import torch.nn.functional as F

class SSMEncoderLayer(nn.Module):
    """
    A simplified State Space Model (SSM) layer for Knowledge Graph relational sequences.
    Inspired by S4/Mamba logic: captures state transitions over a sequence.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.1) # State transition matrix
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.1) # Input matrix
        self.C = nn.Parameter(torch.randn(dim, dim) * 0.1) # Output matrix
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: sequence of entity/relation embeddings (Batch, Seq_Len, Dim)
        """
        # Sequential update: h_t = A * h_t-1 + B * x_t
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.dim, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.matmul(h, self.A) + torch.matmul(x_t, self.B)
            y_t = torch.matmul(h, self.C)
            outputs.append(y_t.unsqueeze(1))
            
        return self.norm(torch.cat(outputs, dim=1))

class KGSSMEncoder(nn.Module):
    """
    Deep SSM-based Knowledge Graph Encoder.
    Uses a stack of SSM layers with residual connections to capture deep structural patterns.
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int = 50, num_layers: int = 2):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        
        # Pile of SSM Layers
        self.layers = nn.ModuleList([SSMEncoderLayer(dim) for _ in range(num_layers)])
        self.fc = nn.Linear(dim, dim)

    def _apply_layers(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Applies stacked SSM layers with residual connections.
        """
        for ssm in self.layers:
            # Residual connection: x = x + ssm(x)
            # This ensures gradients flow well through the 'pile' of layers
            seq = seq + ssm(seq)
        return seq

    def encode_path(self, head_ids: torch.Tensor, rel_ids: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        """
        Full path encoding: [h, r, t]
        """
        h_emb = self.entity_embeddings(head_ids).unsqueeze(1)
        r_emb = self.relation_embeddings(rel_ids).unsqueeze(1)
        t_emb = self.entity_embeddings(tail_ids).unsqueeze(1)
        
        seq = torch.cat([h_emb, r_emb, t_emb], dim=1)
        # Apply deep SSM stack
        ssm_out = self._apply_layers(seq)
        
        return self.fc(ssm_out[:, -1, :])

    def predict_tail(self, head_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Link Prediction encoding: [h, r] -> Predict tail state.
        """
        h_emb = self.entity_embeddings(head_ids).unsqueeze(1)
        r_emb = self.relation_embeddings(rel_ids).unsqueeze(1)
        
        seq = torch.cat([h_emb, r_emb], dim=1)
        # Apply deep SSM stack
        ssm_out = self._apply_layers(seq)
        
        return self.fc(ssm_out[:, -1, :])

    def get_all_embeddings(self) -> torch.Tensor:
        """
        Returns all entity embeddings for retrieval.
        """
        return self.entity_embeddings.weight.data

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device))
