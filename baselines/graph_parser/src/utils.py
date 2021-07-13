import torch

class CharContainer:
    def __init__(self, index_mapping, batch_vocab, voc_lengths):
        self.index_mapping = index_mapping
        self.batch_vocab = batch_vocab
        self.voc_lengths = voc_lengths
        self.vocab_size = len(batch_vocab)

    def to(self, device):
        self.index_mapping = self.index_mapping.to(device)
        self.batch_vocab = self.batch_vocab.to(device)


def dropout_mask(x, size, p):
    return x.new(*size).bernoulli_(1 - p).div_(1 - p)


def concat_per_row(A, B):
    """Concat every row in A with every row in B"""
    m1, n1 = A.shape
    m2, n2 = B.shape

    res = torch.zeros(m1, m2, n1 + n2)
    res[:, :, :n1] = A[:, None, :]
    res[:, :, n1:] = B
    return res.view(m1 * m2, -1)


def batched_concat_per_row(A, B):
    """Concat every row in A with every row in B where
    the first dimension of A and B is the batch dimension"""
    b, m1, n1 = A.shape
    _, m2, n2 = B.shape

    res = torch.zeros(b, m1, m2, n1 + n2)
    res[:, :, :, :n1] = A[:, :, None, :]

    res[:, :, :, n1:] = B[:, None, :, :]
    return res.view(b, m1 * m2, -1)


def create_parameter(*size):
    out = torch.nn.Parameter(
        torch.empty(*size, dtype=torch.float))
    if len(size) > 1:
        torch.nn.init.xavier_uniform_(out)
    else:
        torch.nn.init.uniform_(out)
    return out
