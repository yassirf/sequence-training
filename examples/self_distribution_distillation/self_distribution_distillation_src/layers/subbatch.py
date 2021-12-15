import torch
import torch.nn as nn


class NaiveBatchLayer(nn.Module):
    def __init__(self, cfg, model_list: nn.ModuleList):
        super(NaiveBatchLayer, self).__init__()

        # Number of models
        self.num_models = len(model_list)

        # Collection of separate parallel models
        self.model_list = model_list

    def forward(self, x):
        # At both training inference time we compute the output of all ffns and average
        x = torch.stack([model(x) for model in self.model_list], dim=0)
        return torch.mean(x, dim=0)


class BatchLayer(nn.Module):
    def __init__(self, cfg, model_list: nn.ModuleList):
        super(BatchLayer, self).__init__()

        # Number of models
        self.num_models = len(model_list)

        # Collection of separate parallel models
        self.model_list = model_list

    def forward(self, x):
        """
        This layer splits the batch (dim = 1) into sub batches and passes it through
        separate models
        """

        # Get batch size
        b, f = x.size(1), self.num_models

        # In training mode we split the batch
        if self.training:
            # With mimo-style batch methods we need to assert that the batch size is divisible by
            # the number of ffns in our layer
            assert b % f == 0, "Batch size {} is not divisble by number of ffns {}".format(b, f)

            # Sub batch size which will be passed through each ffn
            sb = b // f

            outputs = []
            for i, model in enumerate(self.model_list):

                # Split the batch (dim = 1) and pass through ffn
                o = model(x[:, i * sb: (i+1) * sb])

                # Append to the outputs
                outputs.append(o)

            # Concatenate over the batch (dim = 1)
            return torch.cat(outputs, dim = 1)

        # At inference time we compute the output of all ffns and average
        x = torch.stack([model(x) for model in self.model_list], dim = 0)
        return torch.mean(x, dim = 0)
