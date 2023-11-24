import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        return torch.tensor(row['userId']), torch.tensor(row['movieId']), torch.tensor(row['rating'])
    

def generate_negative_samples(data, num_items, num_negatives=1):
    """Generate negative samples for each user in the data."""
    data_grouped_by_user = data.groupby('userId').apply(lambda x: x['movieId'].tolist())
    negative_samples = []
    for user_id, positive_items in data_grouped_by_user.items():
        available_items = set(range(num_items)) - set(positive_items)
        negative_items = np.random.choice(list(available_items), size=num_negatives)
        user_ids = [user_id] * num_negatives
        labels = [0] * num_negatives
        negative_samples.extend(zip(user_ids, negative_items, labels))
    negative_samples = pd.DataFrame(negative_samples, columns=['userId', 'movieId', 'rating'])
    data_with_negatives = pd.concat([data, negative_samples])
    return data_with_negatives

def load_data():
    # Load the preprocessed data
    train_data = pd.read_csv('F:/KnowRecDistill/dataset/train.csv')
    val_data = pd.read_csv('F:/KnowRecDistill/dataset/val.csv')
    test_data = pd.read_csv('F:/KnowRecDistill/dataset/test.csv')

    # Combine all data to calculate the number of unique users and items
    all_data = pd.concat([train_data, val_data, test_data])

    # Calculate the number of unique users and items
    num_users = all_data['userId'].nunique()
    num_items = all_data['movieId'].nunique()

    # Generate negative samples for the training and validation data
    train_data = generate_negative_samples(train_data, num_items)
    val_data = generate_negative_samples(val_data, num_items)

    # Convert all positive ratings to 1
    train_data.loc[train_data['rating'] > 0, 'rating'] = 1
    val_data.loc[val_data['rating'] > 0, 'rating'] = 1

    # Create the MovieLensDataset instances
    train_dataset = MovieLensDataset(train_data)
    val_dataset = MovieLensDataset(val_data)
    
    # Create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_dataloader, val_dataloader, num_users, num_items