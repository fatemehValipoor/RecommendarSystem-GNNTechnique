import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split

def read_movielens(raw_dir):
    """Reads the MovieLens dataset from the specified directory and returns three DataFrames."""
    users_df = pd.read_csv(
        os.path.join(raw_dir, "users.dat"),
        sep="::",
        engine='python',
        encoding='ISO-8859-1',
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    )
    movies_df = pd.read_csv(
        os.path.join(raw_dir, "movies.dat"),
        sep="::",
        engine='python',
        encoding='ISO-8859-1',
        names=["MovieID", "Title", "Genres"]
    )
    ratings_df = pd.read_csv(
        os.path.join(raw_dir, "ratings.dat"),
        sep="::",
        engine='python',
        encoding='ISO-8859-1',
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )
    return users_df, movies_df, ratings_df

def build_graph(users_df, movies_df, ratings_df, seed=42):
    """Builds a heterogeneous graph from MovieLens data."""
    num_users = users_df.shape[0]
    num_movies = movies_df.shape[0]

    user_map = {uid: i for i, uid in enumerate(users_df["UserID"])}
    movie_map = {mid: i for i, mid in enumerate(movies_df["MovieID"])}

    ratings_df["user_idx"] = ratings_df["UserID"].map(user_map)
    ratings_df["movie_idx"] = ratings_df["MovieID"].map(movie_map)

    edge_index = torch.tensor([
        ratings_df["user_idx"].values,
        ratings_df["movie_idx"].values
    ], dtype=torch.long)

    edge_attr = torch.tensor(ratings_df["Rating"].values, dtype=torch.float)

    # train/val/test split
    train_idx, test_idx = train_test_split(range(len(ratings_df)), test_size=0.2, random_state=seed)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=seed)

    train_mask = torch.zeros(len(ratings_df), dtype=torch.bool)
    val_mask = torch.zeros(len(ratings_df), dtype=torch.bool)
    test_mask = torch.zeros(len(ratings_df), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = HeteroData()
    data["user"].x = torch.arange(num_users).view(-1, 1).float()
    data["movie"].x = torch.arange(num_movies).view(-1, 1).float()
    data["user", "rates", "movie"].edge_index = edge_index
    data["user", "rates", "movie"].edge_attr = edge_attr
    data["user", "rates", "movie"].train_mask = train_mask
    data["user", "rates", "movie"].val_mask = val_mask
    data["user", "rates", "movie"].test_mask = test_mask

    return data

def process_and_save(dataset_name, raw_dir, processed_dir):
    """Processes dataset and saves HeteroData graph as a .pt file."""
    if dataset_name.lower() == "movielens1m":
        users_df, movies_df, ratings_df = read_movielens(raw_dir)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

    data = build_graph(users_df, movies_df, ratings_df)

    os.makedirs(processed_dir, exist_ok=True)
    save_path = os.path.join(processed_dir, f"{dataset_name}.pt")
    torch.save(data, save_path)
    print(f"{dataset_name} processed dataset saved at {save_path}")
    return save_path
