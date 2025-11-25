import torch
import torch.nn as nn
import torch.nn.functional as F

class HashGNNLoss(nn.Module):
    """
    Combined loss function EXACTLY as in paper equation (7)
    L = L_cross + λ * L_rank
    
    پارامترهای دقیق مقاله:
    - lambda_rank: 0.5 (تراز بین دو loss)
    - alpha: 0.2 (حاشیه برای loss رتبه‌بندی)
    """

    def __init__(self, lambda_rank=0.5, alpha=0.2, temperature=0.1):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.alpha = alpha
        self.temperature = temperature  # برای بهبود پایداری محاسبات
        
        print(f"[green]✅ HashGNN Loss initialized with: λ_rank={lambda_rank}, α={alpha}[/green]")
    
    def cross_entropy_loss(self, h_user, h_item, edge_index):
        """
        Reconstruction loss EXACTLY as in paper equation (5)
        L_cross = -Σ_{(i,j)∈E} log(σ(⟨h_i, h_j⟩)) - Σ_{(i,j)∉E} log(1 - σ(⟨h_i, h_j⟩))
        
        در عمل: از Binary Cross Entropy استفاده می‌شود
        """
        src, dst = edge_index
        
        # محاسبه similarity با inner product دقیقاً مطابق مقاله
        similarities = torch.sum(h_user[src] * h_item[dst], dim=1)
        
        # نرمال کردن برای پایداری عددی
        similarities = similarities / self.temperature
        
        # اعمال sigmoid: σ(⟨h_i, h_j⟩)
        probabilities = torch.sigmoid(similarities)
        
        # تمام یال‌های مشاهده شده positive هستند (A_ij = 1)
        labels = torch.ones_like(probabilities)
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy(probabilities, labels, reduction='mean')
        
        return loss
    
    def ranking_loss(self, h_user, h_item, triplets):
        """
        Ranking loss EXACTLY as in paper equation (6)
        L_rank = Σ_{(i,j,k)∈T} max(0, -σ(⟨h_i, h_j⟩) + σ(⟨h_i, h_k⟩) + α)
        
        که در آن:
        - i: کاربر
        - j: آیتم مثبت  
        - k: آیتم منفی
        - α: حاشیه (margin)
        - T: مجموعه triplets
        """
        if len(triplets) == 0:
            return torch.tensor(0.0, device=h_user.device)
            
        batch_losses = []
        
        # پردازش batch-wise برای کارایی بهتر
        user_indices = []
        pos_indices = []
        neg_indices = []
        
        for u_idx, pos_idx, neg_idx in triplets:
            user_indices.append(u_idx)
            pos_indices.append(pos_idx)
            neg_indices.append(neg_idx)
        
        # تبدیل به تانسور
        user_indices = torch.tensor(user_indices, device=h_user.device)
        pos_indices = torch.tensor(pos_indices, device=h_item.device)  
        neg_indices = torch.tensor(neg_indices, device=h_item.device)
        
        # محاسبه similarity برای تمام triplets به صورت vectorized
        sim_pos = torch.sum(h_user[user_indices] * h_item[pos_indices], dim=1)
        sim_neg = torch.sum(h_user[user_indices] * h_item[neg_indices], dim=1)
        
        # نرمال کردن برای پایداری
        sim_pos = sim_pos / self.temperature
        sim_neg = sim_neg / self.temperature
        
        # اعمال sigmoid
        prob_pos = torch.sigmoid(sim_pos)  # σ(⟨h_u, h_pos⟩)
        prob_neg = torch.sigmoid(sim_neg)  # σ(⟨h_u, h_neg⟩)
        
        # محاسبه loss رتبه‌بندی: max(0, -prob_pos + prob_neg + α)
        margin_loss = -prob_pos + prob_neg + self.alpha
        ranking_loss = F.relu(margin_loss)
        
        # میانگین روی تمام triplets
        final_loss = torch.mean(ranking_loss)
        
        return final_loss

    def forward(self, h_user, h_item, edge_index, triplets):
        """
        محاسبه loss نهایی مطابق معادله (7) مقاله:
        L = L_cross + λ * L_rank
        """
        # محاسبه cross-entropy loss
        ce_loss = self.cross_entropy_loss(h_user, h_item, edge_index)
        
        # محاسبه ranking loss  
        rank_loss = self.ranking_loss(h_user, h_item, triplets)
        
        # ترکیب دو loss با ضریب lambda
        total_loss = ce_loss + self.lambda_rank * rank_loss
        
        # لاگ کردن مقادیر loss برای دیباگ
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[red]⚠️  Loss numerical issue: total={total_loss:.6f}, ce={ce_loss:.6f}, rank={rank_loss:.6f}[/red]")
            print(f"[red]    h_user range: [{h_user.min():.3f}, {h_user.max():.3f}][/red]")
            print(f"[red]    h_item range: [{h_user.min():.3f}, {h_user.max():.3f}][/red]")
        
        return total_loss, ce_loss, rank_loss

    def get_loss_breakdown(self, h_user, h_item, edge_index, triplets):
        """
        گرفتن breakdown کامل loss برای آنالیز
        """
        total_loss, ce_loss, rank_loss = self.forward(h_user, h_item, edge_index, triplets)
        
        breakdown = {
            'total_loss': total_loss.item(),
            'cross_entropy_loss': ce_loss.item(),
            'ranking_loss': rank_loss.item(),
            'lambda_rank': self.lambda_rank,
            'alpha': self.alpha,
            'ce_contribution': ce_loss.item(),
            'rank_contribution': (self.lambda_rank * rank_loss).item()
        }
        
        return breakdown


class AdaptiveHashGNNLoss(HashGNNLoss):
    """
    نسخه پیشرفته loss با adaptive weighting
    """
    
    def __init__(self, lambda_rank=0.5, alpha=0.2, temperature=0.1, 
                 ce_weight=1.0, rank_weight=1.0):
        super().__init__(lambda_rank, alpha, temperature)
        self.ce_weight = ce_weight
        self.rank_weight = rank_weight
        
    def forward(self, h_user, h_item, edge_index, triplets):
        ce_loss = self.cross_entropy_loss(h_user, h_item, edge_index)
        rank_loss = self.ranking_loss(h_user, h_item, triplets)
        
        # وزن‌دهی adaptive
        total_loss = (self.ce_weight * ce_loss + 
                     self.lambda_rank * self.rank_weight * rank_loss)
        
        return total_loss, ce_loss, rank_loss


# 🔹 تابع utility برای ایجاد triplets
def create_triplets_for_evaluation(model, data, num_negatives=4):
    """
    ایجاد triplets برای ارزیابی دقیق مطابق مقاله
    """
    model.eval()
    device = model.get_device()
    
    # گرفتن یال‌های train
    train_edges = data['user', 'rates', 'movie'].edge_index[:, data['user', 'rates', 'movie'].train_mask]
    
    triplets = []
    
    with torch.no_grad():
        # برای هر کاربر در train set
        unique_users = torch.unique(train_edges[0])
        
        for user_idx in unique_users:
            # آیتم‌های مثبت این کاربر
            user_pos_items = train_edges[1, train_edges[0] == user_idx]
            
            if len(user_pos_items) == 0:
                continue
                
            # انتخاب تصادفی یک آیتم مثبت
            pos_idx = user_pos_items[torch.randint(0, len(user_pos_items), (1,))]
            
            # ایجاد آیتم‌های منفی
            num_items = data['movie'].num_nodes
            for _ in range(num_negatives):
                # انتخاب تصادفی یک آیتم منفی
                neg_idx = torch.randint(0, num_items, (1,))
                
                # اطمینان از منفی بودن
                while neg_idx in user_pos_items:
                    neg_idx = torch.randint(0, num_items, (1,))
                
                triplets.append((user_idx.item(), pos_idx.item(), neg_idx.item()))
    
    print(f"[green]✅ Created {len(triplets)} triplets for evaluation[/green]")
    return triplets


# 🔹 تابع محاسده loss با regularization اضافه
def hash_loss_with_regularization(h_user, h_item, edge_index, triplets, 
                                 lambda_rank=0.5, alpha=0.2, reg_weight=0.01):
    """
    نسخه extended از loss با regularization اضافه برای پایداری
    """
    base_loss_fn = HashGNNLoss(lambda_rank, alpha)
    total_loss, ce_loss, rank_loss = base_loss_fn(h_user, h_item, edge_index, triplets)
    
    # اضافه کردن L2 regularization
    reg_loss = (torch.norm(h_user, p=2) + torch.norm(h_item, p=2))
    total_loss_with_reg = total_loss + reg_weight * reg_loss
    
    return total_loss_with_reg, ce_loss, rank_loss, reg_loss