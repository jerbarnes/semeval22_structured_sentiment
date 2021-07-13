import torch
from torch.nn.utils.rnn import pad_sequence
from utils import CharContainer


class PaddedBatch:
    """Container class for padded data"""

    def __init__(self, graph_ids, targetss, unpadding_mask, chars,
                 seq_lengths, indices):
        self.graph_ids = graph_ids
        self.targetss = targetss
        self.unpadding_mask = unpadding_mask
        self.chars = chars
        self.seq_lengths = seq_lengths
        self.indices = indices
        self.sentence_count = len(graph_ids)

    def to(self, device):
        self.targetss = [target.to(device) for target in self.targetss]
        self.unpadding_mask = self.unpadding_mask.to(device)
        self.chars.to(device)
        self.seq_lengths = self.seq_lengths.to(device)
        self.indices = [inds.to(device) for inds in self.indices]

PAD_WORD = (0, 0, 0)


def padded_collate(batch):
    # Sort batch by the longest sequence desc
    batch.sort(key=lambda sequence: len(sequence[3]), reverse=True)

    graph_ids, targetss, char_indices, *index_groups = zip(*batch)

    #print("sens",[(i.shape, j.shape) for i,j in zip(index_groups[0], index_groups[-1])])
    # The number of words in each sequence
    seq_lengths = torch.LongTensor(
        [len(indices) for indices in index_groups[0]])
    max_word_count = seq_lengths[0]

    #print(max_word_count, targets[0].shape, len(heads[0]), len(index_groups[0][0]))
    padded_targetss = [] # when only having a primary (no other) loss
    unpadding_mask = None
    targetss = tuple(zip(*targetss))
    #print(seq_lengths)
    #print(len(targetss[0]))
    #print([x.shape for x in targetss[0]])
    for targets in targetss:
        if not targets[0] is None:
            padded_targets = torch.zeros(
                len(seq_lengths), max_word_count, max_word_count, dtype=torch.long)
            if unpadding_mask is None:
                #unpadding_mask = torch.zeros_like(padded_targets, dtype=torch.uint8)
                unpadding_mask = torch.zeros_like(padded_targets, dtype=torch.bool)
            for i, target in enumerate(targets):
                padded_targets[i, :seq_lengths[i], :seq_lengths[i]] = target
                unpadding_mask[i, :seq_lengths[i], :seq_lengths[i]] = 1
        else:
            padded_targets = None
            unpadding_mask = None
        padded_targetss.append(padded_targets)

    # Batch specific word vocabulary where each word
    # is expressed by its character indices
    batch_voc = list({word for sentence in char_indices for word in sentence})
    batch_voc.append(PAD_WORD)

    batch_voc.sort(key=lambda word: len(word), reverse=True)
    voc_lengths = torch.LongTensor([len(word) for word in batch_voc])
    voc_lookup = {word: i for i, word in enumerate(batch_voc)}
    batch_voc = pad_sequence([torch.LongTensor(tup) for tup in batch_voc],
                             batch_first=True)
    index_mapping = torch.full(
        size=(len(batch), max_word_count),
        fill_value=voc_lookup[PAD_WORD],
        dtype=torch.long)

    # Map each word in the batch to an index in the char word vocabulary
    for i, sentence in enumerate(char_indices):
        for j, word in enumerate(sentence):
            index_mapping[i, j] = voc_lookup[word]

    padded = PaddedBatch(
        graph_ids, padded_targetss,
        unpadding_mask,
        CharContainer(index_mapping, batch_voc, voc_lengths), seq_lengths,
        [pad_sequence(indices, batch_first=True) for indices in index_groups])

    return padded
