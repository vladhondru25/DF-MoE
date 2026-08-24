import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassAnchorMarginLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, margin: float = 1.0, 
                 lambda_repeller: float = 1.0, lambda_orthogonal: float = 0.1):

        super(ClassAnchorMarginLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.lambda_repeller = lambda_repeller
        self.lambda_orthogonal = lambda_orthogonal
        
        

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, anchors) -> torch.Tensor:

        batch_size = embeddings.size(0)
        device = embeddings.device

        target_anchors = anchors[labels] 
        

        loss_attractor = F.mse_loss(embeddings, target_anchors, reduction='mean')
        
        anchor_distances = torch.cdist(anchors, anchors, p=2.0)
        
        repelling_matrix = torch.clamp(self.margin - anchor_distances, min=0.0)
        
        diagonal_mask = torch.eye(self.num_classes, device=device).bool()
        repelling_matrix.masked_fill_(diagonal_mask, 0.0)
        
        denom_repeller = self.num_classes * (self.num_classes - 1)
        loss_repeller = repelling_matrix.sum() / max(denom_repeller, 1)

        anchor_similarity = torch.mm(anchors, anchors.t())
        identity = torch.eye(self.num_classes, device=device)
        loss_orthogonal = F.mse_loss(anchor_similarity, identity, reduction='mean')

        total_loss = (loss_attractor + 
                      self.lambda_repeller * loss_repeller + 
                      self.lambda_orthogonal * loss_orthogonal)
        
        return total_loss