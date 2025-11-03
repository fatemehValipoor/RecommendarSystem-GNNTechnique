import torch
import torch.optim as optim

class HashGNNTrainer:
    """
    Trainer for HashGNN with guidance optimization
    """
    
    def __init__(self, model, loss_fn, optimizer, device, p_init=1.0, p_decay_rate=0.05, p_decay_interval=250):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        # Guidance parameters
        self.p = p_init
        self.p_decay_rate = p_decay_rate
        self.p_decay_interval = p_decay_interval
        
    def train_epoch(self, data, triplet_sampler, epoch):
        """Train for one epoch with guidance strategy"""
        self.model.train()
        
        # Sample triplets for ranking loss
        triplets = triplet_sampler.sample_triplets(num_triplets_per_user=5)
        
        # Update p value (dynamic guidance)
        if epoch > 0 and epoch % self.p_decay_interval == 0:
            self.p = max(0.0, self.p - self.p_decay_rate)
        
        # Forward pass with guidance
        outputs = self.model(data, p=self.p, training=True, use_guidance=True)
        
        # Get only training edges
        train_edge_index = data['user', 'rates', 'movie'].edge_index[
            :, data['user', 'rates', 'movie'].train_mask
        ]
        
        # Calculate loss using guided hash codes
        total_loss, ce_loss, rank_loss = self.loss_fn(
            outputs['h_user_guided'], 
            outputs['h_item_guided'], 
            train_edge_index,
            triplets
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(), 
            'rank_loss': rank_loss.item(),
            'p_value': self.p
        }
    
    def evaluate(self, data, triplet_sampler, mask_type='val'):
        """Evaluate model without guidance"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(data, training=False, use_guidance=False)
            triplets = triplet_sampler.sample_triplets(num_triplets_per_user=5)
            
            # Get appropriate edge mask
            if mask_type == 'val':
                edge_mask = data['user', 'rates', 'movie'].val_mask
            elif mask_type == 'test':
                edge_mask = data['user', 'rates', 'movie'].test_mask
            else:
                edge_mask = data['user', 'rates', 'movie'].train_mask
                
            edge_index = data['user', 'rates', 'movie'].edge_index[:, edge_mask]
            
            total_loss, ce_loss, rank_loss = self.loss_fn(
                outputs['h_user'],
                outputs['h_item'],
                edge_index,
                triplets
            )
            
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'rank_loss': rank_loss.item()
        }