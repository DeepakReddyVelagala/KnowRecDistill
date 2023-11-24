import torch
from torch import nn
from utils import sim

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
        
class FTD(NCF):
    def __init__(self, user_num, item_num, embed_dim, num_layers, dropout=0.5, device='cpu'):
        super().__init__(user_num, item_num, embed_dim, num_layers, dropout)

        self.student_dim = embed_dim

        # Teacher
        self.user_emb_teacher = nn.Embedding(user_num, embed_dim).to(device)
        self.item_emb_teacher = nn.Embedding(item_num, embed_dim).to(device)

        self.teacher_dim = self.user_emb_teacher.weight.size(1)
        
        # Move the embeddings to the device
        self.user_embedding = self.user_embedding.to(device)
        self.item_embedding = self.item_embedding.to(device)

    # topology distillation loss
    def get_TD_loss(self, batch_user, batch_item):
        device = self.user_emb_teacher.weight.device
        batch_user = batch_user.long().to(device)
        batch_item = batch_item.long().to(device)
        s = torch.cat([self.user_embedding(batch_user), self.item_embedding(batch_item)], 0).to(device)
        t = torch.cat([self.user_emb_teacher(batch_user), self.item_emb_teacher(batch_item)], 0).to(device)

        s = s / torch.norm(s, dim=1, keepdim=True)
        t = t / torch.norm(t, dim=1, keepdim=True)

        # Full topology
        t_dist = sim(t, t).view(-1)
        s_dist = sim(s, s).view(-1)  

        # print(f't_dist: {t_dist}, s_dist: {s_dist}')
        total_loss = ((t_dist - s_dist) ** 2).sum() 

        return total_loss

    def forward(self, user_indices, item_indices):
        device = self.user_emb_teacher.weight.device
        user_indices = user_indices.long().to(device)
        item_indices = item_indices.long().to(device)
        user_embedding = self.user_embedding(user_indices).to(device)
        item_embedding = self.item_embedding(item_indices).to(device)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x