import os
import sys
import torch
import pickle
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


def test_hashgnn_movielens():
    """Test HashGNN with full logging"""

    print("\n[bold magenta]🚀 Starting HashGNN Test on MovieLens...[/bold magenta]\n")

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
    # 🚀 Initialize Model
    # ------------------------------------------
    print("\n[cyan]🚀 Initializing HashGNN Model...[/cyan]")

    model = HashGNN(
        num_users=data['user'].num_nodes,
        num_items=data['movie'].num_nodes,
        embedding_dim=64,
        hash_dim=32
    )
    
    print(f"🔧 Model device: {model.get_device()}")
    print(f"📊 Data device: {data['user', 'rates', 'movie'].edge_index.device}")

    print("[green]✅ Model created successfully![/green]")
    print(f"[blue]Model Parameters:[/blue] {sum(p.numel() for p in model.parameters()):,} total parameters")

    # Loss & Optimizer
    loss_fn = HashGNNLoss(lambda_rank=0.5, alpha=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[magenta]🔧 Using device:[/magenta] {device}")

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
        outputs = model(data, training=False, use_guidance=False)   # ← اگر مدل اجازه نده، training=False را حذف کن

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
    # 📈 Evaluation test
    # ------------------------------------------
    print("\n[purple]📈 Running evaluation...[/purple]")

    val_metrics = trainer.evaluate(data, triplet_sampler, mask_type="val")

    print("[green]✅ Evaluation completed[/green]")
    print(val_metrics)

    print("\n[bold green]🎉 All tests passed! HashGNN is ready![/bold green]\n")


if __name__ == "__main__":
    test_hashgnn_movielens()
