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
        # Perform mimo embedding (batch * num-heads, seq, dim * num-heads)
        x = torch.cat([emb(x) for emb in self.embs], dim = -1)

        # The input is of the form (batch * num-heads, seq, dim * num-heads)
        bn, s, dn = x.size()

        # Get the effective batch and vocab size
        b, d = bn//self.num_heads, dn//self.num_heads

        # We need to review the input and choose the relevant ones
        x = x.view(self.num_heads, b, s, self.num_heads, d)

        # Now make a diagonal choice (ensemble, batch, num_classes) this is core to mimo
        x = torch.diagonal(x, offset=0, dim1=0, dim2=3).permute(3, 0, 1, 2)

        # Return the formatted prediction
        return x.reshape(-1, s, d)