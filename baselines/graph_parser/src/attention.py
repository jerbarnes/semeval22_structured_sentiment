import torch
import torch.nn.functional as F
from utils import batched_concat_per_row, create_parameter


class Attention:
    @staticmethod
    def edge_factory(dim, attention_type):
        if attention_type == "bilinear":
            return BilinearEdgeAttention(dim)
        elif attention_type == "biaffine":
            return BiaffineEdgeAttention(dim)
        elif attention_type == "affine":
            return AffineEdgeAttention(dim)
        else:
            raise Exception("{attention_type} is not a valid attention type".format(attention_type))

    @staticmethod
    def label_factory(dim, n_labels, attention_type):
        if attention_type == "bilinear":
            return BilinearLabelAttention(dim, n_labels)
        elif attention_type == "biaffine":
            return BiaffineLabelAttention(dim, n_labels)
        elif attention_type == "affine":
            return AffineLabelAttention(dim, n_labels)
        else:
            raise Exception("{attention_type} is not a valid attention type".format(attention_type))

    def get_label_scores(self, head, dep):
        # head, dep: [sequence x batch x mlp]
        raise NotImplementedError()

    def get_edge_scores(self, head, dep):
        # head, dep: [sequence x batch x mlp]
        raise NotImplementedError()


class BilinearEdgeAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_U = create_parameter(dim, dim)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]

        # (batch x seq x seq)
        return torch.einsum("bij,jk,bok->bio", (head, self.edge_U, dep))


class BilinearLabelAttention(torch.nn.Module):
    def __init__(self, dim, n_labels):
        super().__init__()
        self.label_U_diag = create_parameter(n_labels, dim)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]

        # (batch x label x seq x seq)
        return torch.einsum("bij,lj,boj->blio", (head, self.label_U_diag, dep))


class BiaffineEdgeAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_U = create_parameter(dim, dim)
        self.edge_W = create_parameter(1, 2 * dim)
        self.edge_b = create_parameter(1)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)

        # (batch x seq x seq)
        t1 = torch.einsum("bij,jk,bok->bio", (head, self.edge_U, dep))

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (1 x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x 1 x seq*seq)
        t2 = self.edge_W @ concated.transpose(1, 2)

        # (batch x 1 x seq*seq) => (batch x seq x seq)
        t2 = t2.view(batch_size, sequence_size, sequence_size)

        return t1 + t2 + self.edge_b


class BiaffineLabelAttention(torch.nn.Module):
    def __init__(self, dim, n_labels):
        super().__init__()

        self.label_U_diag = create_parameter(n_labels, dim)
        self.label_W = create_parameter(n_labels, 2 * dim)
        self.label_b = create_parameter(n_labels)
        self.n_labels = n_labels

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)

        # (batch x label x seq x seq)
        t1 = torch.einsum("bij,lj,boj->blio", (head, self.label_U_diag, dep))

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (labels x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x labels x seq*seq)
        t2 = self.label_W @ concated.transpose(1, 2)

        # (batch x labels x seq*seq) => (batch x labels x seq x seq)
        t2 = t2.view(batch_size, self.n_labels, sequence_size, sequence_size)

        return t1 + t2 + self.label_b[None, :, None, None]


class AffineLabelAttention(torch.nn.Module):
    def __init__(self, dim, n_labels):
        super().__init__()

        self.label_W = create_parameter(n_labels, 2 * dim)
        self.label_b = create_parameter(n_labels)
        self.n_labels = n_labels

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (labels x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x labels x seq*seq)
        t2 = self.label_W @ concated.transpose(1, 2)

        # (batch x labels x seq*seq) => (batch x labels x seq x seq)
        t2 = t2.view(batch_size, self.n_labels, sequence_size, sequence_size)

        return t2 + self.label_b[None, :, None, None]

class AffineEdgeAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_W = create_parameter(1, 2 * dim)
        self.edge_b = create_parameter(1)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (1 x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x 1 x seq*seq)
        t2 = self.edge_W @ concated.transpose(1, 2)

        # (batch x 1 x seq*seq) => (batch x seq x seq)
        t2 = t2.view(batch_size, sequence_size, sequence_size)

        return t2 + self.edge_b



class DotProductAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dk = dim ** 0.5#torch.sqrt(dk)

    def forward(self, attention_matrix, output):
        # TODO really dim=1?
        attention_matrix = attention_matrix
        am = F.softmax(attention_matrix.transpose(-2,-1) * self.dk, dim=1) @ output
        return am

