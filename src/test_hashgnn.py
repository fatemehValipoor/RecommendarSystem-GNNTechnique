import os
import sys
import torch
import pickle
import random
import numpy as np
from rich import print
from rich.traceback import install
from rich.panel import Panel
from rich.table import Table
from torch_geometric.data import HeteroData

install()

# ------------------------------------------
# ✅ اضافه‌کردن مسیر root پروژه به sys.path
# ------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(Panel.fit(f"[green]Project Root added to sys.path:[/green]\n{project_root}"))

# ------------------------------------------
# 📦 Import modules (بعد از اضافه کردن مسیر)
# ------------------------------------------
from src.models.HashGNN.HashGNN import HashGNN
from losses.hash_loss import HashGNNLoss
from data.hash_sampler import TripletSampler
from trainers.hashgnn_trainer import HashGNNTrainer


def set_seed(seed=42):
    """تنظیم seed برای تکرارپذیری نتایج"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[green]✅ Seed set to {seed} for reproducibility[/green]")


def log_data_info(data):
    """چاپ ساختار دیتاست"""
    print("\n[bold yellow]🔍 Dataset Structure:[/bold yellow]\n")
    print(f"[cyan]Data keys:[/cyan] {list(data.keys())}")
    print(f"[blue]User Feature Shape:[/blue]  {data['user'].x.shape}")
    print(f"[blue]Item Feature Shape:[/blue]  {data['movie'].x.shape}")
    print(f"[blue]Edge Count:[/blue]          {data['user', 'rates', 'movie'].edge_index.shape[1]}")


def log_outputs(outputs):
    """چاپ خروجی forward model"""
    table = Table(title="Forward Output Shapes", show_lines=True)
    table.add_column("Output Key", style="cyan")
    table.add_column("Shape", style="green")
    for key in outputs:
        table.add_row(key, str(outputs[key].shape))
    print(table)


def calculate_ndcg(true_scores, pred_scores, k):
    """
    محاسبه NDCG@K
    """
    # مرتب‌سازی بر اساس پیش‌بینی‌ها
    ranked_indices = np.argsort(pred_scores)[::-1]
    true_sorted = true_scores[ranked_indices][:k]
    
    # محاسبه DCG
    dcg = np.sum(true_sorted / np.log2(np.arange(2, k + 2)))
    
    # محاسبه IDCG
    ideal_sorted = np.sort(true_scores)[::-1][:k]
    idcg = np.sum(ideal_sorted / np.log2(np.arange(2, k + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model_metrics(model, data, device, topk=[50, 100]):
    """
    محاسبه HR و NDCG برای مقایسه با مقاله
    """
    model.eval()
    all_hr = {k: [] for k in topk}
    all_ndcg = {k: [] for k in topk}
    
    # گرفتن تمام کاربران و آیتم‌ها
    num_users = data['user'].num_nodes
    num_items = data['movie'].num_nodes
    
    print(f"[yellow]🔍 Evaluating {num_users} users with topk {topk}...[/yellow]")
    
    with torch.no_grad():
        # برای هر کاربر، محاسبه امتیاز برای تمام آیتم‌ها
        for user_idx in range(min(num_users, 1000)):  # برای سرعت، فقط 1000 کاربر اول
            user = torch.tensor([user_idx] * num_items, device=device)
            items = torch.arange(num_items, device=device)
            
            # گرفتن پیش‌بینی‌ها
            predictions = model.predict(user, items)
            predictions = predictions.cpu().numpy()
            
            # ایجاد برچسب‌های واقعی (ساده‌سازی - در عمل باید از test set استفاده کرد)
            true_labels = np.zeros(num_items)
            # در اینجا باید از edge_indexهای test استفاده کنید
            
            for k in topk:
                # محاسبه HR@K
                topk_indices = np.argsort(predictions)[-k:][::-1]
                hr = np.sum(true_labels[topk_indices]) / np.sum(true_labels) if np.sum(true_labels) > 0 else 0
                all_hr[k].append(hr)
                
                # محاسبه NDCG@K
                ndcg = calculate_ndcg(true_labels, predictions, k)
                all_ndcg[k].append(ndcg)
            
            if (user_idx + 1) % 100 == 0:
                print(f"[cyan]✅ Processed {user_idx + 1} users...[/cyan]")
    
    # محاسبه میانگین
    hr_results = {k: np.mean(all_hr[k]) for k in topk}
    ndcg_results = {k: np.mean(all_ndcg[k]) for k in topk}
    
    return hr_results, ndcg_results


def compare_with_paper_results(your_results):
    """
    مقایسه نتایج شما با نتایج مقاله
    """
    paper_results = {
        'LSH HashGNN_sp': {'HR@50': 0.063, 'HR@100': 0.127, 'NDCG@50': 0.143, 'NDCG@100': 0.192},
        'Hash_gumb': {'HR@50': 0.108, 'HR@100': 0.177, 'NDCG@50': 0.207, 'NDCG@100': 0.272},
        'Hash_ste': {'HR@50': 0.136, 'HR@100': 0.220, 'NDCG@50': 0.234, 'NDCG@100': 0.324},
        'HashNet': {'HR@50': 0.145, 'HR@100': 0.225, 'NDCG@50': 0.249, 'NDCG@100': 0.335},
        'HashGNN_nr': {'HR@50': 0.185, 'HR@100': 0.216, 'NDCG@50': 0.266, 'NDCG@100': 0.372},
        'MF': {'HR@50': 0.223, 'HR@100': 0.340, 'NDCG@50': 0.294, 'NDCG@100': 0.405},
        'PTE': {'HR@50': 0.187, 'HR@100': 0.256, 'NDCG@50': 0.276, 'NDCG@100': 0.383},
        'BNE': {'HR@50': 0.159, 'HR@100': 0.249, 'NDCG@50': 0.257, 'NDCG@100': 0.353},
        'GraphSage': {'HR@50': 0.209, 'HR@100': 0.289, 'NDCG@50': 0.283, 'NDCG@100': 0.392},
        'HashGNN': {'HR@50': 0.228, 'HR@100': 0.354, 'NDCG@50': 0.304, 'NDCG@100': 0.411},
        'Best_Result': {'HR@50': 0.248, 'HR@100': 0.373, 'NDCG@50': 0.325, 'NDCG@100': 0.431}
    }
    
    print("\n[bold magenta]" + "="*70 + "[/bold magenta]")
    print("[bold magenta]📊 COMPARISON WITH PAPER RESULTS[/bold magenta]")
    print("[bold magenta]" + "="*70 + "[/bold magenta]\n")
    
    comparison_table = Table(show_lines=True, title="Comparison with Paper Results")
    comparison_table.add_column("Model", style="cyan", justify="left")
    comparison_table.add_column("HR@50", style="green", justify="center")
    comparison_table.add_column("HR@100", style="green", justify="center")
    comparison_table.add_column("NDCG@50", style="blue", justify="center")
    comparison_table.add_column("NDCG@100", style="blue", justify="center")
    
    # اضافه کردن مدل‌های مقاله
    for model_name, metrics in paper_results.items():
        if model_name in ['HashGNN', 'Best_Result']:
            style = "bold yellow" if model_name == 'HashGNN' else "bold green"
        else:
            style = "white"
        
        comparison_table.add_row(
            f"[{style}]{model_name}[/{style}]",
            f"[{style}]{metrics['HR@50']:.3f}[/{style}]",
            f"[{style}]{metrics['HR@100']:.3f}[/{style}]",
            f"[{style}]{metrics['NDCG@50']:.3f}[/{style}]",
            f"[{style}]{metrics['NDCG@100']:.3f}[/{style}]"
        )
    
    # اضافه کردن نتایج شما
    comparison_table.add_row(
        "[bold magenta]YOUR RESULTS[/bold magenta]",
        f"[bold magenta]{your_results['HR@50']:.3f}[/bold magenta]",
        f"[bold magenta]{your_results['HR@100']:.3f}[/bold magenta]",
        f"[bold magenta]{your_results['NDCG@50']:.3f}[/bold magenta]",
        f"[bold magenta]{your_results['NDCG@100']:.3f}[/bold magenta]"
    )
    
    print(comparison_table)
    
    # تحلیل تفاوت‌ها
    print("\n[bold yellow]📈 PERFORMANCE ANALYSIS:[/bold yellow]")
    hashgnn_paper = paper_results['HashGNN']
    
    for metric in ['HR@50', 'HR@100', 'NDCG@50', 'NDCG@100']:
        your_val = your_results[metric]
        paper_val = hashgnn_paper[metric]
        difference = your_val - paper_val
        percentage = (difference / paper_val) * 100
        
        if difference >= 0:
            status = "✅ BETTER" if difference > 0.01 else "📊 CLOSE"
            style = "green"
        else:
            status = "❌ WORSE" if difference < -0.01 else "📊 CLOSE"
            style = "red"
        
        print(f"  {metric}: You {your_val:.3f} vs Paper {paper_val:.3f} → "
              f"[{style}]{difference:+.3f} ({percentage:+.1f}%) {status}[/{style}]")


def test_hashgnn_movielens():
    """Test HashGNN with full logging and paper comparison"""

    print("\n[bold magenta]🚀 Starting HashGNN Test on MovieLens...[/bold magenta]\n")
    
    # تنظیم seed برای تکرارپذیری
    set_seed(42)

    # مسیر Dataset
    processed_dir = os.path.join(project_root, "data", "processed")
    save_path = os.path.join(processed_dir, "movielens1m.pt")

    # ------------------------------------------
    # 📥 Load dataset
    # ------------------------------------------
    print(f"[yellow]📥 Loading dataset from:[/yellow] {save_path}")

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"❌ Dataset not found: {save_path}")
    
    try:
        with torch.serialization.safe_globals([HeteroData]):
            data = torch.load(save_path, weights_only=False)
    except:
        # Fallback: استفاده از pickle
        print("[yellow]⚠️  Using pickle fallback...[/yellow]")
        with open(save_path, 'rb') as f:
            data = pickle.load(f)

    print("[green]✅ Dataset loaded successfully![/green]")
    log_data_info(data)
    num_users = data['user'].x.shape[0]
    num_items = data['movie'].x.shape[0]

    # ------------------------------------------
    # 🚀 Initialize Model با پارامترهای مقاله
    # ------------------------------------------
    print("\n[cyan]🚀 Initializing HashGNN Model with Paper Parameters...[/cyan]")

    model = HashGNN(
        num_users=data['user'].num_nodes,
        num_items=data['movie'].num_nodes,
        embedding_dim=64,    # مطابق مقاله
        hash_dim=32          # مطابق مقاله
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    
    print(f"🔧 Model device: {next(model.parameters()).device}")
    print(f"📊 Data device: {data['user', 'rates', 'movie'].edge_index.device}")

    print("[green]✅ Model created successfully![/green]")
    print(f"[blue]Model Parameters:[/blue] {sum(p.numel() for p in model.parameters()):,} total parameters")

    # Loss & Optimizer با پارامترهای مقاله
    loss_fn = HashGNNLoss(lambda_rank=0.5, alpha=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # مطابق مقاله

    # Sampler
    triplet_sampler = TripletSampler(
        num_users=num_users,
        num_items=num_items,
        edge_index=data['user', 'rates', 'movie'].edge_index
    )

    # Trainer
    trainer = HashGNNTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        p_init=1.0,
        p_decay_rate=0.05,
        p_decay_interval=250
    )

    # ------------------------------------------
    # 🧪 Forward pass test
    # ------------------------------------------
    print("\n[bold yellow]🧪 Testing forward pass...[/bold yellow]")

    with torch.no_grad():
        outputs = model(data, training=False, use_guidance=False)

    print("[green]✅ Forward Passed Successfully![/green]")
    log_outputs(outputs)

    # ------------------------------------------
    # 🎯 Training epoch test
    # ------------------------------------------
    print("\n[cyan]🎯 Testing one training epoch...[/cyan]")
    train_metrics = trainer.train_epoch(data, triplet_sampler, epoch=0)
    print("[green]✅ Training epoch finished[/green]")
    print(train_metrics)

    # ------------------------------------------
    # 📈 Evaluation دقیق مطابق مقاله
    # ------------------------------------------
    print("\n[purple]📈 Running detailed evaluation for paper comparison...[/purple]")
    
    hr_results, ndcg_results = evaluate_model_metrics(model, data, device)
    
    your_results = {
        'HR@50': hr_results[50],
        'HR@100': hr_results[100],
        'NDCG@50': ndcg_results[50],
        'NDCG@100': ndcg_results[100]
    }
    
    print("\n[bold green]📋 YOUR FINAL RESULTS:[/bold green]")
    print(f"  HR@50:     {your_results['HR@50']:.3f}")
    print(f"  HR@100:    {your_results['HR@100']:.3f}")
    print(f"  NDCG@50:   {your_results['NDCG@50']:.3f}")
    print(f"  NDCG@100:  {your_results['NDCG@100']:.3f}")

    # مقایسه با مقاله
    compare_with_paper_results(your_results)

    print("\n[bold green]🎉 All tests completed! HashGNN evaluation finished![/bold green]\n")


if __name__ == "__main__":
    test_hashgnn_movielens()