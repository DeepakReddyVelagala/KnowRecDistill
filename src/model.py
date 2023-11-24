import torch
from torch import nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers, dropout=0.5):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.fc_layers, output_dim = self._create_layers(embed_dim, num_layers, dropout)
        self.output_layer = nn.Linear(output_dim, 1)

    def _create_layers(self, embed_dim, num_layers, dropout):
        fc_layers = []
        input_dim = 2 * embed_dim  # Adjust the input size
        for i in range(num_layers - 1):
            output_dim = input_dim // 2
            fc_layers.append(nn.Linear(input_dim, output_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            input_dim = output_dim  # Update the input size for the next layer
        return nn.Sequential(*fc_layers), output_dim

    def forward(self, user_indices, item_indices):
        user_indices = user_indices.long()
        item_indices = item_indices.long()
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x
