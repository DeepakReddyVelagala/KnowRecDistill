import torch
from model import FTD, NCF
from dataset import load_data
import numpy as np
from utils import calculate_precision_recall

def ftd_train_epoch(model, train_loader, criterion, optimizer, device, lmbda_TD):
    model.train()
    total_loss = 0.0

    for i, (user, item, label) in enumerate(train_loader):
        user = user.to(device)
        item = item.to(device)
        label = label.to(device)

        # Forward pass
        output = model(user, item)
        output = output.view(-1)

        # Compute L_base
        base_loss = criterion(output, label)

        # Topology Distillation
        # Compute L_FTD
        TD_loss = model.get_TD_loss(user.unique(), item.unique())

        # Compute L = L_base + lambda * L_FTD
        loss = base_loss + TD_loss * lmbda_TD

        # Update Student model by minimizing L
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def evaluate_ftd_epoch(model, test_loader, criterion, device, lmbda_TD, top_n):
    model.eval()
    total_loss = 0.0
    precisions = []
    recalls = []

    with torch.no_grad():
        for i, (user, item, label) in enumerate(test_loader):
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)

            # Forward pass
            output = model(user, item)
            output = output.view(-1)

            # Compute L_base
            base_loss = criterion(output, label)

            # Topology Distillation
            # Compute L_FTD
            TD_loss = model.get_TD_loss(user.unique(), item.unique())

            # Compute L = L_base + lambda * L_FTD
            loss = base_loss + TD_loss * lmbda_TD
            # print(TD_loss)
            total_loss += loss.item()

            # Calculate the top-N recommendations
            top_n = min(top_n, output.numel())
            _, top_indices = torch.topk(output, top_n)

            # Calculate the precision and recall
            precision, recall = calculate_precision_recall(top_indices, label, top_n)
            precisions.append(precision)
            recalls.append(recall)

    # Calculate the accuracy
    accuracy = np.mean(precisions)

    return total_loss, accuracy, np.mean(precisions), np.mean(recalls)

def ftd_train(train_dataloader, val_dataloader,user_emb_teacher, item_emb_teacher, num_users, num_items, device, model_name, num_layers=2):
    # Define the hyperparameters
    embed_dim = 50
    num_epochs = 2
    top_n = 10
    lmbda_TD = 0.5

    # Instantiate the mode
    model = FTD(num_users, num_items, embed_dim, num_layers, device=device)
    model = model.to(device)

    # Load the pretrained embeddings
    model.user_emb_teacher.weight.data = user_emb_teacher
    model.item_emb_teacher.weight.data = item_emb_teacher

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train and validate the model
    for epoch in range(num_epochs):
        train_loss = ftd_train_epoch(model, train_dataloader, criterion, optimizer, device, lmbda_TD)
        test_loss, accuracy, precision, recall = evaluate_ftd_epoch(model, val_dataloader, criterion, device, lmbda_TD, top_n)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
        print(f'Precision@{top_n}: {precision:.4f}')
        print(f'Recall@{top_n}: {recall:.4f}')

    # Save the model
    torch.save(model.state_dict(), model_name+'.pth')

def main():
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train_dataloader, val_dataloader, num_users, num_items = load_data()
    
    # Load the pretrained model
    pretrained_model = NCF(num_users, num_items, 50, 2)
    pretrained_model.load_state_dict(torch.load('F:\KnowRecDistill\save_models\model.pth'))
    pretrained_model = pretrained_model.to(device)

    # Get the user and item embeddings
    user_emb_teacher = pretrained_model.user_embedding.weight.data.to(device)
    item_emb_teacher = pretrained_model.item_embedding.weight.data.to(device)


    # Train FTD
    ftd_train(train_dataloader, val_dataloader, user_emb_teacher, item_emb_teacher, num_users, num_items, device, model_name='model_ftd', num_layers=2)

if __name__ == "__main__":
    main()