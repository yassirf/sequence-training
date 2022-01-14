import torch
import torch.nn as nn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx = padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class MimoEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads, padding_idx):
        super(MimoEmbedding, self).__init__()

        # Assign parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.padding_idx = padding_idx

        # Create the embedding model
        self.embs = nn.ModuleList([
            Embedding(num_embeddings, embedding_dim, padding_idx) for _ in range(num_heads)
        ])

    def forward(self, x):

        if self.training:
            # Reshape the input for mimo embedding (heads, batch, len)
            x = x.view(self.num_heads, x.size(0)//self.num_heads, x.size(1))
        else:
            # In inference we repeat the example a number of times
            x = x.unsqueeze(0).repeat(self.num_heads, 1, 1)

        # Perform mimo embedding (batch, len, dim)
        x = sum(emb(x[i]) for i, emb in enumerate(self.embs))
        return x