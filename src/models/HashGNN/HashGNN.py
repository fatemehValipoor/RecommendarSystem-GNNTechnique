import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod  
    def backward(ctx, grad_output):
        return grad_output.clone()

class HashGNNLayer(nn.Module):
    """
    Graph Convolutional Layer با پیاده‌سازی دستی برای گراف bipartite
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x_target, x_source, edge_index):
        """
        پیاده‌سازی دستی معادله (1) مقاله
        x_target: ویژگی‌های نودهای هدف
        x_source: ویژگی‌های نودهای مبدأ  
        edge_index: [2, num_edges] - edge_index[0] = source, edge_index[1] = target
        """
        # تجمیع دستی همسایه‌ها
        source_idx, target_idx = edge_index
        
        # ایجاد ماتریس aggregation
        num_target = x_target.size(0)
        num_source = x_source.size(0)
        
        # میانگین‌گیری همسایه‌ها برای هر نود هدف
        agg_matrix = torch.zeros(num_target, num_source, device=x_target.device)
        agg_matrix[target_idx, source_idx] = 1.0
        
        # نرمالایز کردن (میانگین)
        degree = agg_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # جلوگیری از تقسیم بر صفر
        agg_matrix = agg_matrix / degree
        
        # تجمیع همسایه‌ها
        neigh_agg = torch.mm(agg_matrix, x_source)
        
        # ترکیب خود نود و همسایه‌ها: MEAN{خود ∪ همسایه}
        combined = torch.stack([x_target, neigh_agg], dim=1)
        combined_mean = torch.mean(combined, dim=1)
        
        # تبدیل خطی + فعال‌ساز
        out = self.lin(combined_mean)
        out = F.relu(out)
        
        return out

class HashGNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hash_dim=32):
        super().__init__()
        
        # امبدینگ اولیه
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # لایه‌های GCN
        self.gcn1 = HashGNNLayer(embedding_dim, 128)
        self.gcn2 = HashGNNLayer(128, 64)
        
        # لایه هش
        self.hash_layer = nn.Sequential(
            nn.Linear(64, hash_dim),
            nn.Tanh()
        )
        
        self.num_users = num_users
        self.num_items = num_items
        self.hash_dim = hash_dim
        
    def forward(self, data, p=0.5, training=True, use_guidance=True):
        # 🔹 مدیریت device
        device = next(self.parameters()).device
        
        # 🔹 درست کردن اندیس‌ها
        user_indices = torch.arange(self.num_users, device=device)
        item_indices = torch.arange(self.num_items, device=device)
        
        x_user = self.user_emb(user_indices)  # [num_users, 64]
        x_item = self.item_emb(item_indices)  # [num_items, 64]
        
        # 🔹 انتقال edge_index به device مناسب
        edge_index_ui = data['user', 'rates', 'movie'].edge_index.to(device)  # user->movie
        
        print(f"🔍 Debug: user features shape: {x_user.shape}")
        print(f"🔍 Debug: item features shape: {x_item.shape}")
        print(f"🔍 Debug: edge_index_ui shape: {edge_index_ui.shape}")
        
        # 🔹 **PROPAGATION دستی و درست**
        
        # لایه 1: کاربران از فیلم‌ها یاد می‌گیرند 
        # source = items, target = users
        print("🔍 Starting user propagation (items → users)...")
        x_user_1 = self.gcn1(
            x_target=x_user,      # هدف: کاربران
            x_source=x_item,      # مبدأ: فیلم‌ها  
            edge_index=edge_index_ui[[1, 0]]  # معکوس: movie->user
        )
        print(f"🔍 Debug: x_user_1 shape: {x_user_1.shape}")
        
        # لایه 1: فیلم‌ها از کاربران یاد می‌گیرند
        # source = users, target = items  
        print("🔍 Starting item propagation (users → items)...")
        x_item_1 = self.gcn1(
            x_target=x_item,      # هدف: فیلم‌ها
            x_source=x_user,      # مبدأ: کاربران
            edge_index=edge_index_ui  # user->movie
        )
        print(f"🔍 Debug: x_item_1 shape: {x_item_1.shape}")
        
        # لایه 2: ادامه propagation
        print("🔍 Starting layer 2...")
        x_user_2 = self.gcn2(
            x_target=x_user_1,
            x_source=x_item_1, 
            edge_index=edge_index_ui[[1, 0]]  # movie->user
        )
        x_item_2 = self.gcn2(
            x_target=x_item_1,
            x_source=x_user_1,
            edge_index=edge_index_ui  # user->movie
        )
        
        print(f"🔍 Debug: x_user_2 shape: {x_user_2.shape}")
        print(f"🔍 Debug: x_item_2 shape: {x_item_2.shape}")
        
        # لایه هش
        z_user = self.hash_layer(x_user_2)
        z_item = self.hash_layer(x_item_2)
        
        # 🔹 **استفاده از SignSTE**
        h_user = SignSTE.apply(z_user)
        h_item = SignSTE.apply(z_item)
        
        print("✅ Forward pass completed successfully!")
        
        # Guidance strategy
        if training and use_guidance:
            p_tensor = torch.tensor(p, device=device)
            
            Q_user = torch.bernoulli(torch.full_like(z_user, p_tensor))
            Q_item = torch.bernoulli(torch.full_like(z_item, p_tensor))
            
            h_user_guided = Q_user * z_user + (1 - Q_user) * h_user
            h_item_guided = Q_item * z_item + (1 - Q_item) * h_item
            
            return {
                'z_user': z_user, 'z_item': z_item,
                'h_user': h_user, 'h_item': h_item,
                'h_user_guided': h_user_guided, 'h_item_guided': h_item_guided,
                'user_embeddings': x_user_2,
                'item_embeddings': x_item_2
            }
        else:
            return {
                'z_user': z_user, 'z_item': z_item,
                'h_user': h_user, 'h_item': h_item,
                'user_embeddings': x_user_2,
                'item_embeddings': x_item_2
            }

    def get_device(self):
        return next(self.parameters()).device