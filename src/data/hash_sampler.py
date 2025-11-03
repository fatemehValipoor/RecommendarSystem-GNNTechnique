import torch
import numpy as np
from collections import defaultdict

class TripletSampler:
    """
    Sampler for triplets EXACTLY as described in paper section 3.4
    """
    
    def __init__(self, num_users, num_items, edge_index):
        self.num_users = num_users
        self.num_items = num_items
        self.edge_index = edge_index
        self.user_pos_items = self._build_user_pos_items()
        
    def _build_user_pos_items(self):
        """Build dictionary of positive items for each user"""
        user_pos_items = defaultdict(list)
        src, dst = self.edge_index
        
        for u, i in zip(src.tolist(), dst.tolist()):
            user_pos_items[u].append(i)
            
        return user_pos_items
    
    def sample_triplets(self, num_triplets_per_user=5):
        """
        Sample triplets for ranking loss
        Returns: list of (user_idx, positive_item_idx, negative_item_idx)
        """
        triplets = []
        
        for u in range(self.num_users):
            if u not in self.user_pos_items:
                continue
                
            pos_items = self.user_pos_items[u]
            
            for _ in range(num_triplets_per_user):
                if len(pos_items) == 0:
                    continue
                    
                # Sample positive item from user's interactions
                pos_idx = np.random.choice(pos_items)
                
                # Sample negative item that user hasn't interacted with
                neg_idx = np.random.randint(0, self.num_items)
                while neg_idx in pos_items:
                    neg_idx = np.random.randint(0, self.num_items)
                    
                triplets.append((u, pos_idx, neg_idx))
                
        return triplets