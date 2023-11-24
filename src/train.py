import torch
import pandas as pd
from dataset import MovieLensDataset
from torch.utils.data import DataLoader
from model import NCF
import torch.nn as nn
import numpy as np

def calculate_precision_recall(top_indices, labels, top_n):
    relevant_items = set(labels.nonzero().view(-1).cpu().numpy())
    recommended_items = set(top_indices.cpu().numpy())
    intersection = relevant_items & recommended_items
    precision = len(intersection) / len(recommended_items)
    recall = len(intersection) / len(relevant_items) if len(relevant_items) > 0 else 0
    return precision, recall

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

def train(model, dataloader, optimizer, criterion, device):
    print('Training the model...')
    model.train()
    total_loss = 0
    # i = 0
    for user_indices, item_indices, labels in dataloader:
        # i += 1
        # print(i)
        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(user_indices, item_indices)
        outputs = outputs.view(-1)

        # Ensure labels is a 1D tensor with the same shape as outputs
        labels = labels.view(-1)
        # print(f'Labels min: {labels.min()}, max: {labels.max()}')
        # print(f'Outputs min: {outputs.min()}, max: {outputs.max()}')
        # Calculate the loss
        loss = criterion(outputs, labels)
        # print(f'Loss: {loss.item()}')
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, top_n, device):
    model.eval()
    precisions = []
    recalls = []
    with torch.no_grad():
        for user_indices, item_indices, labels in dataloader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            labels = labels.to(device)

            # Forward pass
            
            outputs = model(user_indices, item_indices)
            outputs = outputs.view(-1)
            # print(f'Size of outputs: {outputs.numel()}, top_n: {top_n}')
            # Calculate the top-N recommendations
            top_n = min(top_n, outputs.numel())
            _, top_indices = torch.topk(outputs, top_n)

            # Calculate the precision and recall
            precision, recall = calculate_precision_recall(top_indices, labels, top_n)
            precisions.append(precision)
            recalls.append(recall)

            # print(f'Precision: {precision}, Recall: {recall}')

    return np.mean(precisions), np.mean(recalls)

def main():
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train_dataloader, val_dataloader, num_users, num_items = load_data()

    # Define the hyperparameters
    embed_dim = 50
    num_layers = 2
    num_epochs = 5
    top_n = 10

    # Instantiate the model, the criterion, and the optimizer
    model = NCF(num_users, num_items, embed_dim, num_layers)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())


    # Train and validate the model
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        precision, recall = validate(model, val_dataloader, top_n, device)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Precision@{top_n}: {precision:.4f}')
        print(f'Recall@{top_n}: {recall:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()