import torch
import torch.nn as nn
from .utils import mimo_batchify

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None: nn.init.constant_(m.weight[padding_idx], 0)
    return m


class MimoEmbedding(nn.Module):
    def __init__(self, num_inputs, num_embeddings, embedding_dim, padding_idx = None):
        """
        Multi-input Embedding layer, which takes in a collection of sequences and returns a mix
        Note that supplying a padding_idx does not ensure it is trainable
        """
        super(MimoEmbedding, self).__init__()

        self.num_inputs = num_inputs
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Create embedding layers and initialise properly
        self.embs = nn.ModuleList([
            Embedding(num_embeddings, embedding_dim, padding_idx) for _ in range(num_inputs)
        ])

    def forward(self, x):
        """
        The input should have a shape (num * batch, len) and
        The return will have shape (batch, len, dim)
        """

        # Batchify input (num, batch, len)
        x = mimo_batchify(x, self.num_inputs)

        return sum(emb(x[i]) for i, emb in enumerate(self.embs))

