import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import  WeightedRandomSampler
from dataset import MRIDataset
from train import train_model
from evaluate import evaluate_model
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
import random
import h5py
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from model import model
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Add this before torch operations
def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepyNet3D on DCE-MRI breast cancer dataset")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--input_mode", type=str, default="delta",choices=["delta"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="runs/")
    parser.add_argument("--model", type=str, default="model")
    parser.add_argument('--use_h5', action='store_true', help='Use preprocessed .h5 file instead of .nii.gz')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保卷积算法确定性
    torch.backends.cudnn.benchmark = True

def compute_class_weights(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = torch.tensor([total / counter[i] for i in sorted(counter.keys())], dtype=torch.float32)
    return weights
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file=None
    def _init_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images = self.h5_file['images']
            self.labels = self.h5_file['labels']
            self.ids = self.h5_file['ids']
    def __len__(self):
        self._init_file()
        return len(self.labels)
    def __getitem__(self, idx):
        self._init_file()
        x = torch.tensor(self.images[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
    def __del__(self):
        try:
            if hasattr(self,'h5_file') and self.h5_file is not None:
                self.h5_file.close()
        except:
                pass
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LABELS_CSV = r""
    df = pd.read_csv(LABELS_CSV).dropna(subset=["pcr"])
    labels=df['pcr'].tolist()
    if args.use_h5:
        full_dataset = H5Dataset("MRIDataset.h5")
        labels=full_dataset.labels
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)),labels)):
        print(f"\n🔁 Fold {fold+1}/5")
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        full_dataset.augment = False
        train_subset.dataset.augment = True
        train_labels = [train_subset[i][1].item() for i in range(len(train_subset))]
        class_weights = compute_class_weights(train_labels).to(device)
        weights = 1. / class_weights
        sample_weights = [weights[int(lbl)] for lbl in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        num_works = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=num_works)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=num_works)
        model=model().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        run_dir = os.path.join(args.save_dir, f"{args.input_mode}/{args.model}/{fold+1}")
        os.makedirs(run_dir, exist_ok=True)
        best_model_path = os.path.join(run_dir, "best_model.pt")

        print("🚀 Training...")
        train_model(
            model=model,
            loaders=(train_loader, val_loader),
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_epochs=args.epochs,
            patience=args.patience,
            save_path=best_model_path
        )
        print("🔍 Evaluating...")
        model.load_state_dict(torch.load(best_model_path))
        metrics = evaluate_model(model,val_loader, device, save_path=os.path.join(run_dir, "metrics.json"))
        all_metrics.append(metrics)
if __name__ == "__main__":
    main()
