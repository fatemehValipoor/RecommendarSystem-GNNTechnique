import torch
import torch.nn as nn
import torch.nn.functional as F

class HashGNNLoss(nn.Module):
    """
    Combined loss function EXACTLY as in paper equation (7)
    L = L_cross + λ * L_rank
    """
    
    def __init__(self, lambda_rank=0.5, alpha=0.2):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.alpha = alpha
    
    def cross_entropy_loss(self, h_user, h_item, edge_index):
        """
        Reconstruction loss EXACTLY as in paper equation (5)
        """
        src, dst = edge_index
        
        # Calculate inner products: ⟨h_i, h_j⟩
        similarities = torch.sum(h_user[src] * h_item[dst], dim=1)
        
        # Apply sigmoid: σ(⟨h_i, h_j⟩)
        probabilities = torch.sigmoid(similarities)
        
        # All observed links are positive (A_ij = 1)
        labels = torch.ones_like(probabilities)
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy(probabilities, labels)
        return loss
    
    def ranking_loss(self, h_user, h_item, triplets):
        """
        Ranking loss EXACTLY as in paper equation (6)
        """
        if len(triplets) == 0:
            return torch.tensor(0.0, device=h_user.device)
            
        losses = []
        for u_idx, pos_idx, neg_idx in triplets:
            # Similarity with positive item: σ(⟨h_u, h_pos⟩)
            sim_pos = torch.sigmoid(torch.dot(h_user[u_idx], h_item[pos_idx]))
            
            # Similarity with negative item: σ(⟨h_u, h_neg⟩)  
            sim_neg = torch.sigmoid(torch.dot(h_user[u_idx], h_item[neg_idx]))
            
            # Ranking loss: max(0, -sim_pos + sim_neg + α)
            loss = F.relu(-sim_pos + sim_neg + self.alpha)
            losses.append(loss)
            
        return torch.mean(torch.stack(losses))
    
    def forward(self, h_user, h_item, edge_index, triplets):
        ce_loss = self.cross_entropy_loss(h_user, h_item, edge_index)
        rank_loss = self.ranking_loss(h_user, h_item, triplets)
        total_loss = ce_loss + self.lambda_rank * rank_loss
        
        return total_loss, ce_loss, rank_loss