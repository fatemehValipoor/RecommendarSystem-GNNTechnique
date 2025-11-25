import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import ndcg_score

class HashGNNTrainer:
    """
    Trainer for HashGNN with guidance optimization and paper evaluation metrics
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
        
        # برای ذخیره بهترین مدل بر اساس NDCG
        self.best_ndcg = 0.0
        self.best_model_state = None
        
    def calculate_ndcg(self, true_scores, pred_scores, k):
        """
        محاسبه NDCG@K مطابق مقاله
        """
        if len(true_scores) == 0:
            return 0.0
            
        # مرتب‌سازی بر اساس پیش‌بینی‌ها
        ranked_indices = np.argsort(pred_scores)[::-1]
        true_sorted = true_scores[ranked_indices][:k]
        
        # محاسبه DCG
        dcg = np.sum(true_sorted / np.log2(np.arange(2, k + 2)))
        
        # محاسبه IDCG
        ideal_sorted = np.sort(true_scores)[::-1][:k]
        idcg = np.sum(ideal_sorted / np.log2(np.arange(2, k + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0

    def calculate_hr_ndcg_metrics(self, data, mask_type='test', topk=[50, 100]):
        """
        محاسبه HR و NDCG برای مقایسه با مقاله
        """
        self.model.eval()
        all_hr = {k: [] for k in topk}
        all_ndcg = {k: [] for k in topk}
        
        # گرفتن ماسک مناسب
        if mask_type == 'val':
            edge_mask = data['user', 'rates', 'movie'].val_mask
        elif mask_type == 'test':
            edge_mask = data['user', 'rates', 'movie'].test_mask
        else:
            edge_mask = data['user', 'rates', 'movie'].train_mask
            
        # گرفتن یال‌های تست
        test_edges = data['user', 'rates', 'movie'].edge_index[:, edge_mask]
        
        # ایجاد دیکشنری برای کاربران و آیتم‌های مثبت
        user_positives = {}
        for i in range(test_edges.shape[1]):
            user = test_edges[0, i].item()
            item = test_edges[1, i].item()
            if user not in user_positives:
                user_positives[user] = []
            user_positives[user].append(item)
        
        num_users = data['user'].num_nodes
        num_items = data['movie'].num_nodes
        
        print(f"[yellow]🔍 Evaluating {len(user_positives)} users with topk {topk}...[/yellow]")
        
        with torch.no_grad():
            processed_users = 0
            for user_idx, pos_items in user_positives.items():
                if processed_users >= 1000:  # برای سرعت، فقط 1000 کاربر اول
                    break
                    
                user_tensor = torch.tensor([user_idx] * num_items, device=self.device)
                items_tensor = torch.arange(num_items, device=self.device)
                
                # گرفتن پیش‌بینی‌ها - اگر تابع predict وجود ندارد از forward استفاده کنید
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(user_tensor, items_tensor)
                else:
                    # استفاده از forward معمولی
                    user_emb = self.model.user_embedding(user_tensor)
                    item_emb = self.model.item_embedding(items_tensor)
                    predictions = torch.sum(user_emb * item_emb, dim=1)
                
                predictions = predictions.cpu().numpy()
                
                # ایجاد برچسب‌های واقعی
                true_labels = np.zeros(num_items)
                for pos_item in pos_items:
                    if pos_item < num_items:
                        true_labels[pos_item] = 1
                
                for k in topk:
                    # محاسبه HR@K
                    topk_indices = np.argsort(predictions)[-k:][::-1]
                    hr = np.sum(true_labels[topk_indices]) / len(pos_items) if len(pos_items) > 0 else 0
                    all_hr[k].append(hr)
                    
                    # محاسبه NDCG@K
                    ndcg = self.calculate_ndcg(true_labels, predictions, k)
                    all_ndcg[k].append(ndcg)
                
                processed_users += 1
                if processed_users % 100 == 0:
                    print(f"[cyan]✅ Processed {processed_users} users...[/cyan]")
        
        # محاسبه میانگین
        hr_results = {k: np.mean(all_hr[k]) for k in topk}
        ndcg_results = {k: np.mean(all_ndcg[k]) for k in topk}
        
        return hr_results, ndcg_results

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
        
        # Gradient clipping برای پایداری آموزش (مطابق مقاله)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        metrics = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(), 
            'rank_loss': rank_loss.item(),
            'p_value': self.p
        }
        
        # ارزیابی دوره‌ای هر 10 epoch
        if epoch % 10 == 0:
            hr_results, ndcg_results = self.calculate_hr_ndcg_metrics(data, mask_type='val', topk=[50, 100])
            metrics.update({
                'HR@50': hr_results[50],
                'HR@100': hr_results[100],
                'NDCG@50': ndcg_results[50],
                'NDCG@100': ndcg_results[100]
            })
            
            # ذخیره بهترین مدل بر اساس NDCG@100
            current_ndcg = ndcg_results[100]
            if current_ndcg > self.best_ndcg:
                self.best_ndcg = current_ndcg
                self.best_model_state = {
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'ndcg': current_ndcg
                }
                print(f"[green]🎯 New best model saved at epoch {epoch} with NDCG@100: {current_ndcg:.4f}[/green]")
        
        return metrics
    
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
            
            # محاسبه متریک‌های HR و NDCG
            hr_results, ndcg_results = self.calculate_hr_ndcg_metrics(data, mask_type=mask_type)
            
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'rank_loss': rank_loss.item(),
            'HR@50': hr_results[50],
            'HR@100': hr_results[100],
            'NDCG@50': ndcg_results[50],
            'NDCG@100': ndcg_results[100]
        }

    def evaluate_final(self, data, mask_type='test'):
        """
        ارزیابی نهایی با بهترین مدل ذخیره شده
        """
        if self.best_model_state is not None:
            print("[yellow]🔄 Loading best model for final evaluation...[/yellow]")
            self.model.load_state_dict(self.best_model_state['model_state'])
        
        print(f"[bold magenta]📊 FINAL EVALUATION on {mask_type.upper()} set[/bold magenta]")
        
        metrics = self.evaluate(data, None, mask_type=mask_type)  # نیازی به sampler نیست
        
        print("\n[bold green]📋 FINAL RESULTS:[/bold green]")
        print(f"  HR@50:     {metrics['HR@50']:.3f}")
        print(f"  HR@100:    {metrics['HR@100']:.3f}")
        print(f"  NDCG@50:   {metrics['NDCG@50']:.3f}")
        print(f"  NDCG@100:  {metrics['NDCG@100']:.3f}")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        
        return metrics

    def get_paper_comparison(self, final_metrics):
        """
        مقایسه نتایج نهایی با مقاله
        """
        paper_results = {
            'HashGNN': {
                'HR@50': 0.228, 'HR@100': 0.354, 
                'NDCG@50': 0.304, 'NDCG@100': 0.411
            },
            'Best_Result': {
                'HR@50': 0.248, 'HR@100': 0.373, 
                'NDCG@50': 0.325, 'NDCG@100': 0.431
            }
        }
        
        print("\n[bold yellow]📈 COMPARISON WITH PAPER:[/bold yellow]")
        for metric in ['HR@50', 'HR@100', 'NDCG@50', 'NDCG@100']:
            your_val = final_metrics[metric]
            paper_val = paper_results['HashGNN'][metric]
            diff = your_val - paper_val
            status = "✅ ABOVE" if diff > 0.01 else "📊 CLOSE" if abs(diff) <= 0.01 else "❌ BELOW"
            color = "green" if diff > 0.01 else "yellow" if abs(diff) <= 0.01 else "red"
            
            print(f"  {metric}: You {your_val:.3f} vs Paper {paper_val:.3f} → "
                  f"[{color}]{diff:+.3f} {status}[/{color}]")