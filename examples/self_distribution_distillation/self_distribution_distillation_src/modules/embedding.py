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
            Embedding(num_embeddings, embedding_dim, padding_idx)
        for _ in range(num_heads)])

    def get_embeddings(self, x):
        return torch.cat([emb(x) for emb in self.embs], dim = -1)

    def reformat_output(self, x):
        # The input is of the form (batch * num-heads, len, dim * num-heads)
        bn, s, dn = x.size()

        # Get the effective batch and vocab size
        b, d = bn//self.num_heads, dn//self.num_heads

        # We need to review the input and choose the relevant ones
        x = x.view(self.num_heads, b, s, self.num_heads, d)

        # Now make a diagonal choice (ensemble, batch, num_classes) this is core to mimo
        x = torch.diagonal(x, offset=0, dim1=0, dim2=3).permute(3, 0, 1, 2)

        # Return the formatted prediction (num-heads, batch, len, dim)
        return x

    def forward_embeddings_inference(self, x):
        # Get the embeddings for the batch using different embeddings (batch, len, num-heads, dim)
        # This is equivalent to repeating the input a number of times
        x = self.get_embeddings(x)

        # Return the weighted sum of the input tokens / sum of embeddings
        return x.reshape(x.size(0), x.size(1), self.num_heads, -1).sum(dim = 2)

    def forward_embeddings_train(self, x):
        # Get size batch and len dimensions for future use
        bn, s = x.size()

        # In training mode we need to create weighted averages of sequences for each sentence
        # Since there are already predefined variables for each sentence length care has to be taken
        embeddings = self.get_embeddings(x)

        # This has the shape (num-heads, batch, len, dim)
        embeddings = self.reformat_output(embeddings)

        augmentations = [embeddings]
        # This process needs to be repeated to generate augmentations for mimo
        for i in range(1, self.num_heads):
            aug = self.get_embeddings(x[torch.randperm(bn)])

            # This now has a size of (num-heads, batch, len, dim)
            aug = self.reformat_output(aug)

            # Ensure a rotation of the augmentation of step i in dimension 0
            aug = torch.roll(aug, i, 0)

            # Append the augmentations
            augmentations.append(aug)

        # Now return the sum of these embeddings
        embeddings = sum(augmentations).reshape(bn, s, -1)

        return embeddings

    def forward(self, x):
        if self.training:
            return self.forward_embeddings_train(x)
        return self.forward_embeddings_inference(x)