import pandas as pd
import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        return torch.tensor(row['userId']), torch.tensor(row['movieId']), torch.tensor(row['rating'])