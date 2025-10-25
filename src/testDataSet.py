import os
import sys
import torch
from rich import print
from rich.traceback import install

install()
# اضافه کردن فولدر data به مسیر پایتون برای import درست
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))

from make_dataset import process_and_save

if __name__ == "__main__":
    # مسیر پروژه فعلی (یک سطح بالاتر از src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # مسیر داده‌های خام و پردازش شده نسبت به فولدر پروژه
    raw_dir = os.path.join(project_root, "data", "raw", "ml-1m")
    processed_dir = os.path.join(project_root, "data", "processed")
    dataset_name = "movielens1m"

    if not os.path.exists(raw_dir):
        print("Raw dataset not found.")

    print("raw_dir:", raw_dir)
    print("processed_dir:", processed_dir)

    # پردازش و ذخیره گراف
    save_path = process_and_save(dataset_name, raw_dir, processed_dir)

    # بارگذاری فایل ذخیره شده با weights_only=False برای PyTorch >=2.6
    data = torch.load(save_path, weights_only=False)

    # چاپ خلاصه گراف
    print(data)

    # بررسی ویژگی‌ها و ماسک‌ها
    print("Number of users:", data['user'].x.shape[0])
    print("Number of movies:", data['movie'].x.shape[0])
    print("Number of edges:", data['user', 'rates', 'movie'].edge_index.shape[1])
    print("Edge ratings sample:", data['user', 'rates', 'movie'].edge_attr[:5])
    print("Train mask sum:", data['user', 'rates', 'movie'].train_mask.sum())
    print("Validation mask sum:", data['user', 'rates', 'movie'].val_mask.sum())
    print("Test mask sum:", data['user', 'rates', 'movie'].test_mask.sum())
