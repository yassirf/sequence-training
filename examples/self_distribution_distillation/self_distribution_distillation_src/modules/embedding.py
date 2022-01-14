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
        self.embs = Embedding(num_embeddings, embedding_dim * num_heads, padding_idx)

        # Dimensionality reduction
        self.reduction = nn.Linear(embedding_dim * num_heads, embedding_dim)

    def forward(self, x):
        # Get the embeddings for each sub-batch (num-heads * batch, len, num-heads * dim)
        x = self.embs(x)

        # In training mode we need to permute the embeddings so inputs share features
        if self.training:
            # Get the current size of the input
            bn, s, dn = x.size()

            # Now we reshape the input
            x = x.view(self.num_heads, bn//self.num_heads, s, self.num_heads, dn//self.num_heads)

            # Transpose the input so each input shares features with remaining inputs
            x = torch.transpose(x, 0, 3).view(bn, s, dn)

        # Perform dimensionality reduction to match the required size of the network
        x = self.reduction(x)

        return x