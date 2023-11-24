from model import NCF
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Load the ratings data
ratings_df = pd.read_csv('F:/KnowRecDistill/dataset/ratings.csv')

# Load the movies data
movies_df = pd.read_csv('F:/KnowRecDistill/dataset/movies.csv')

# Create a mapping from original movie IDs to new IDs
unique_movie_ids = ratings_df['movieId'].unique()
movie_to_idx = {original_id: idx for idx, original_id in enumerate(unique_movie_ids)}
idx_to_movie = {idx: original_id for original_id, idx in movie_to_idx.items()}

# Apply the mapping to the movie IDs in the ratings DataFrame
ratings_df['movieId'] = ratings_df['movieId'].apply(lambda x: movie_to_idx[x])

# Calculate the number of unique users and items
num_users = ratings_df['userId'].nunique()
num_items = ratings_df['movieId'].nunique()

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NCF(num_users, num_items, embed_dim=50, num_layers=2)
model.load_state_dict(torch.load(r'F:/KnowRecDistill\save_models\model.pth'))
model = model.to(device)
model.eval()

########################### Only modify the user_id ###########################
user_id = 1
###############################################################################

# Get the first five movies the user already rated
user_ratings = ratings_df[ratings_df['userId'] == user_id]
first_five_rated = user_ratings.head()
first_five_rated_movies = movies_df[movies_df['movieId'].isin(first_five_rated['movieId'])]
print('First five rated movies:')
print(first_five_rated_movies['title'])

# Create a DataLoader with all items for the selected user
all_items = torch.tensor(ratings_df['movieId'].unique())
user_indices = torch.full_like(all_items, user_id)

# Check if user_id is within range
assert user_id < num_users, f"user_id ({user_id}) is out of range (num_users: {num_users})"

# Check if all item indices are within range
assert all_items.max().item() < num_items, f"An item index is out of range (max item index: {all_items.max().item()}, num_items: {num_items})"

data = list(zip(user_indices, all_items))
dataloader = DataLoader(data, batch_size=64)

# Make predictions for all items
predictions = []
for user_indices, item_indices in dataloader:
    user_indices = user_indices.to(device)
    item_indices = item_indices.to(device)
    output = model(user_indices, item_indices)
    predictions.extend(output.detach().cpu())

predictions = torch.tensor(predictions)
# print("Unique items:", torch.unique(all_items))
# print("Unique predictions:", torch.unique(predictions))

# Convert predictions to tensor
predictions_tensor = torch.FloatTensor(predictions)

if predictions_tensor.shape[0] >= 10:
    # Get the indices of the top 10 predictions
    top_indices = torch.topk(predictions_tensor, 10).indices

    # Get the top 10 items
    top_items = all_items[top_indices]
    # print(f'Top 10 recommended items for user {user_id}: {top_items}')
    top_movies = [idx_to_movie[idx.item()] for idx in top_items]
    top_movies_df = movies_df[movies_df['movieId'].isin(top_movies)]
    print('Top 10 recommended movies:')
    print(top_movies_df['title'])
else:
    print(f'Not enough predictions to get top 10 items for user {user_id}')