import torch
from dataset import load_data
from model import NCF
import numpy as np
from utils import calculate_precision_recall

def train_ncf_epoch(model, dataloader, optimizer, criterion, device):
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

def validate_ncf_epoch(model, dataloader, top_n, device):
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

def train_ncf(train_dataloader, val_dataloader, num_users, num_items, device, model_name, num_layers=2):
    # Define the hyperparameters
    embed_dim = 50
    num_epochs = 5
    top_n = 10

    # Instantiate the model, the criterion, and the optimizer
    model = NCF(num_users, num_items, embed_dim, num_layers)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())


    # Train and validate the model
    for epoch in range(num_epochs):
        train_loss = train_ncf_epoch(model, train_dataloader, optimizer, criterion, device)
        precision, recall = validate_ncf_epoch(model, val_dataloader, top_n, device)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Precision@{top_n}: {precision:.4f}')
        print(f'Recall@{top_n}: {recall:.4f}')

    # Save the model
    torch.save(model.state_dict(), model_name+'.pth')


def main():
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train_dataloader, val_dataloader, num_users, num_items = load_data()

    # train ncf
    train_ncf(train_dataloader, val_dataloader, num_users, num_items, device, model_name='model_ncf', num_layers=2)

if __name__ == "__main__":
    main()