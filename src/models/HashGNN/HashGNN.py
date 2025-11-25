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
        
        # 🔹 پارامترهای دقیق مطابق مقاله
        self.embedding_dim = embedding_dim
        self.hash_dim = hash_dim
        self.num_users = num_users
        self.num_items = num_items
        
        # 🔹 امبدینگ اولیه با مقداردهی مطابق مقاله
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # 🔹 لایه‌های GCN مطابق معماری مقاله
        self.gcn1 = HashGNNLayer(embedding_dim, 128)
        self.gcn2 = HashGNNLayer(128, 64)
        
        # 🔹 لایه هش با معماری دقیق مقاله
        self.hash_layer = nn.Sequential(
            nn.Linear(64, hash_dim),
            nn.Tanh()  # مطابق مقاله از Tanh استفاده می‌شود
        )
        
        # 🔹 لایه پیش‌بینی برای امتیازدهی
        self.predict_layer = nn.Linear(hash_dim, 1, bias=False)
        
        # 🔹 مقداردهی اولیه وزن‌ها مطابق مقاله
        self._init_weights()
        
    def _init_weights(self):
        """مقداردهی اولیه وزن‌ها مطابق مقاله با Xavier initialization"""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.gcn1.lin.weight)
        nn.init.zeros_(self.gcn1.lin.bias)
        nn.init.xavier_uniform_(self.gcn2.lin.weight)
        nn.init.zeros_(self.gcn2.lin.bias)
        nn.init.xavier_uniform_(self.hash_layer[0].weight)
        nn.init.zeros_(self.hash_layer[0].bias)
        nn.init.xavier_uniform_(self.predict_layer.weight)
        
        print(f"[green]✅ Model weights initialized with Xavier uniform[/green]")
    
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
        
        # 🔹 **PROPAGATION دستی و درست**
        
        # لایه 1: کاربران از فیلم‌ها یاد می‌گیرند 
        # source = items, target = users
        x_user_1 = self.gcn1(
            x_target=x_user,      # هدف: کاربران
            x_source=x_item,      # مبدأ: فیلم‌ها  
            edge_index=edge_index_ui[[1, 0]]  # معکوس: movie->user
        )
        
        # لایه 1: فیلم‌ها از کاربران یاد می‌گیرند
        # source = users, target = items  
        x_item_1 = self.gcn1(
            x_target=x_item,      # هدف: فیلم‌ها
            x_source=x_user,      # مبدأ: کاربران
            edge_index=edge_index_ui  # user->movie
        )
        
        # لایه 2: ادامه propagation
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
        
        # لایه هش
        z_user = self.hash_layer(x_user_2)
        z_item = self.hash_layer(x_item_2)
        
        # 🔹 **استفاده از SignSTE**
        h_user = SignSTE.apply(z_user)
        h_item = SignSTE.apply(z_item)
        
        # Guidance strategy مطابق مقاله
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

    def predict(self, user_indices, item_indices):
        """
        تابع پیش‌بینی برای ارزیابی HR و NDCG
        user_indices: تانسور اندیس کاربران [batch_size]
        item_indices: تانسور اندیس آیتم‌ها [batch_size]
        """
        # گرفتن هش کدهای کاربران و آیتم‌ها
        device = next(self.parameters()).device
        
        # اگر مدل در حالت آموزش است، از حالت evaluation استفاده کن
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # گرفتن تمام امبدینگ‌ها
            all_user_emb = self.user_emb(torch.arange(self.num_users, device=device))
            all_item_emb = self.item_emb(torch.arange(self.num_items, device=device))
            
            # propagation (می‌توانید کش کنید برای سرعت)
            # برای سادگی، از امبدینگ‌های مستقیم استفاده می‌کنیم
            user_emb = all_user_emb[user_indices]
            item_emb = all_item_emb[item_indices]
            
            # محاسبه similarity (دات پروداکت)
            predictions = torch.sum(user_emb * item_emb, dim=1)
            
        if was_training:
            self.train()
            
        return predictions

    def get_similarity(self, h_user, h_item):
        """
        محاسبه similarity بین هش کدهای کاربر و آیتم
        مطابق مقاله از inner product استفاده می‌شود
        """
        return torch.sum(h_user * h_item, dim=1)

    def get_device(self):
        return next(self.parameters()).device

    def get_hash_codes(self, data, users=None, items=None):
        """
        گرفتن هش کدهای نهایی برای کاربران و آیتم‌ها
        """
        self.eval()
        device = self.get_device()
        
        with torch.no_grad():
            outputs = self.forward(data, training=False, use_guidance=False)
            
            if users is not None:
                user_codes = outputs['h_user'][users]
            else:
                user_codes = outputs['h_user']
                
            if items is not None:
                item_codes = outputs['h_item'][items]
            else:
                item_codes = outputs['h_item']
                
        return user_codes, item_codes

# 🔹 تابع کمکی برای محاسبه متریک‌ها
def calculate_recommendation_metrics(model, data, topk=100):
    """
    محاسبه متریک‌های توصیه‌گر به صورت batch برای کارایی بهتر
    """
    model.eval()
    device = model.get_device()
    
    num_users = data['user'].num_nodes
    num_items = data['movie'].num_items
    
    # گرفتن تمام هش کدها
    user_codes, item_codes = model.get_hash_codes(data)
    
    # محاسبه similarity ماتریس کامل
    similarity_matrix = torch.mm(user_codes, item_codes.t())  # [num_users, num_items]
    
    # گرفتن آیتم‌های مثبت از داده‌های تست
    test_edges = data['user', 'rates', 'movie'].edge_index[:, data['user', 'rates', 'movie'].test_mask]
    
    hr_scores = []
    ndcg_scores = []
    
    for user_idx in range(num_users):
        # آیتم‌های مثبت برای این کاربر
        user_pos_items = test_edges[1, test_edges[0] == user_idx]
        
        if len(user_pos_items) == 0:
            continue
            
        # گرفتن امتیازهای این کاربر
        user_scores = similarity_matrix[user_idx]
        
        # محاسبه HR@k
        _, topk_indices = torch.topk(user_scores, topk)
        hit = torch.any(torch.isin(topk_indices, user_pos_items))
        hr_scores.append(hit.float().item())
        
        # محاسبه NDCG@k (ساده‌سازی شده)
        relevance = torch.zeros(num_items, device=device)
        relevance[user_pos_items] = 1
        ranked_relevance = relevance[topk_indices]
        
        dcg = torch.sum(ranked_relevance / torch.log2(torch.arange(2, topk + 2, device=device)))
        idcg = torch.sum(torch.sort(relevance, descending=True)[0][:topk] / 
                        torch.log2(torch.arange(2, topk + 2, device=device)))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg.item())
    
    avg_hr = torch.mean(torch.tensor(hr_scores)).item()
    avg_ndcg = torch.mean(torch.tensor(ndcg_scores)).item()
    
    return avg_hr, avg_ndcg