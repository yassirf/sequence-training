

def mimo_batchify(x, num_heads):
    """
    For some tensor x of size (num-heads * batch, ...)
    reshape it into a form factor that is matched with mimo systems.
    """

    # Ensure the batch is divisible by number of heads
    assert x.size(0) % num_heads == 0

    # Get the shape for arbitrary sized tensors
    shape = list(x.size())

    # Now reshape
    return x.reshape(num_heads, -1, *shape[1:])


def mimo_convert_to_list(batch):
    return [batch[i] for i in range(batch.size(0))]