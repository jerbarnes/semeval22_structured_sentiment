import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate
import os
import torchtext
from collections import defaultdict
import json
from convert_to_bio import to_bio
import re
import numpy as np

class Vocab(defaultdict):
    def __init__(self, train=True):
        super().__init__(lambda: len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]
        self.idx2w = self.update_idx2w()

    def set_vocab(self):
        self.train = False

    def train(self):
        self.train = True

    def update_idx2w(self):
        self.idx2w = dict([(i, w) for w, i in self.items()])

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return torch.tensor([self[w] for w in ws], dtype=torch.long)
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        #idxs = set(idx2w.keys())
        #return [idx2w[int(i)] if int(i) in idxs else "UNK" for i in ids]
        return [self.idx2w[int(i)] for i in ids]


class SetVocab(dict):
    def __init__(self, vocab):
        self.update(vocab)

    def ws2ids(self, ws):
        return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]


class Label2Idx():
    def __init__(self):
        super().__init__()
        self.label2idx = {"sources": {"O": 0,
                                      "B-holder": 1,
                                      "I-holder": 2
                                      },
                          "targets": {"O": 0,
                                      "B-targ-Positive": 1,
                                      "B-targ-Negative": 1,
                                      "B-targ-Neutral": 1,
                                      "B-targ-None": 1,
                                      "I-targ-Positive": 2,
                                      "I-targ-Negative": 2,
                                      "I-targ-Neutral": 2,
                                      "I-targ-None": 2
                                      },
                          "expressions": {"O": 0,
                                          "B-exp-Positive": 1,
                                          "B-exp-Negative": 2,
                                          "B-exp-Neutral": 3,
                                          "B-exp-None": 3,
                                          "I-exp-Positive": 4,
                                          "I-exp-Negative": 5,
                                          "I-exp-Neutral": 6,
                                          "I-exp-None": 6
                                          }
                          }
        self.idx2label = {"sources": {0: "O", 1: "B-holder", 2: "I-holder"},
                          "targets": {0: "O", 1: "B-targ", 2: "I-targ"},
                          "expressions": {0: "O",
                                          1: "B-exp-Positive",
                                          2: "B-exp-Negative",
                                          3: "B-exp-Neutral",
                                          4: "I-exp-Positive",
                                          5: "I-exp-Negative",
                                          6: "I-exp-Neutral"}
                          }
    def labels2idxs(self, labels, annotation="sources"):
        return [self.label2idx[annotation][label] for label in labels]
    #
    def idxs2labels(self, idxs, annotation="sources"):
        return [self.idx2label[annotation][idx] for idx in idxs]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class RelationSplit(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_padded_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item: len(item[1]), reverse=True)

        sent_ids = [sent_id for sent_id, w, e1, e2, label in batch]
        words = pack_sequence([w for sent_id, w, e1, e2, label in batch])
        e1s = pack_sequence([e1 for sent_id, w, e1, e2, label in batch])
        e2s = pack_sequence([e2 for sent_id, w, e1, e2, label in batch])
        targets = default_collate([t for sent_id, w, e1, e2, t in batch])

        return sent_ids, words, e1s, e2s, targets

class RelationDataset(object):
    def __init__(self, vocab, lower_case):

        self.vocab = vocab
        self.splits = {}
        self.labels = [0, 1]

    def open_split(self, data_file, lower_case):
        sent_id = torchtext.data.Field(sequential=False)
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        e1 = torchtext.data.Field(batch_first=True)
        e2 = torchtext.data.Field(batch_first=True)
        label = torchtext.data.Field(sequential=False)
        data = torchtext.data.TabularDataset(data_file, format="json", fields={"sent_id": ("sent_id", sent_id), "text": ("text", text), "e1": ("e1", e1), "e2": ("e2", e2), "label": ("label", label)})
        data_split = [(item.sent_id,
                       torch.LongTensor(self.vocab.ws2ids(item.text)),
                       torch.LongTensor(item.e1),
                       torch.LongTensor(item.e2),
                       torch.LongTensor([int(item.label)])) for item in data]
        return data_split

    def get_split(self, filename, lower_case=True):
        return RelationSplit(self.open_split(filename, lower_case))


class Split(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
        sequences = [w for _, w, _ in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[2], sorted_batch)), batch_first=True, padding_value=0)
        sent_ids = list(map(lambda x: x[0], sorted_batch))
        return sent_ids, sequences_padded, lengths, labels


class ExtractionDataset(object):
    def __init__(self, vocab, label2idx, lower_case):
        #
        self.vocab = vocab
        self.label2idx = label2idx
        #
    def open_split(self, data_file, lower_case, annotation="sources"):
        sent_id = torchtext.data.Field(sequential=False)
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        sources = torchtext.data.Field(sequential=True)
        targets = torchtext.data.Field(sequential=True)
        expressions = torchtext.data.Field(sequential=True)
        data = torchtext.data.TabularDataset(data_file, format="json", fields={"sent_id": ("sent_id", sent_id), "text": ("text", text), "sources": ("sources", sources), "targets": ("targets", targets), "expressions": ("expressions", expressions)})
        if annotation == "sources":
            data_split = [(item.sent_id,
                           torch.LongTensor(self.vocab.ws2ids(item.text)),
                           torch.LongTensor(self.label2idx.labels2idxs(item.sources, annotation="sources"))) for item in data]
        elif annotation == "targets":
            data_split = [(item.sent_id,
                           torch.LongTensor(self.vocab.ws2ids(item.text)),
                           torch.LongTensor(self.label2idx.labels2idxs(item.targets, annotation="targets"))) for item in data]
        elif annotation == "expressions":
            data_split = [(item.sent_id,
                           torch.LongTensor(self.vocab.ws2ids(item.text)),
                           torch.LongTensor(self.label2idx.labels2idxs(item.expressions, annotation="expressions"))) for item in data]
        else:
            data_split = [(item.sent_id,
                           torch.LongTensor(self.vocab.ws2ids(item.text)),
                           torch.LongTensor(self.label2idx.labels2idxs(item.sources, annotation="sources")),
                           torch.LongTensor(self.label2idx.labels2idxs(item.targets, annotation="targets")),
                           torch.LongTensor(self.label2idx.labels2idxs(item.expressions, annotation="expressions"))) for item in data]
        return data_split
    #
    def get_split(self, filename, lower_case=True, annotation="sources"):
        return Split(self.open_split(filename, lower_case, annotation=annotation))

class ExtractionInferenceDataset(object):
    def __init__(self, vocab, label2idx, lower_case):
        #
        self.vocab = vocab
        self.label2idx = label2idx
        #
    def open_split(self, data_file, lower_case, annotation="sources"):
        sent_id = torchtext.data.Field(sequential=False)
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        with open(data_file) as infile:
            data = json.load(infile)
        data_split = [(item["sent_id"],
                       torch.LongTensor(self.vocab.ws2ids(item["text"].split())),
                       torch.zeros(len(item["text"].split()))) for item in data]
        return data_split
    #
    def get_split(self, filename, lower_case=True, annotation="sources"):
        return Split(self.open_split(filename, lower_case, annotation=annotation))

def get_best_run(weightdir):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_acc = 0.0
    best_weights = ''
    for file in os.listdir(weightdir):
        if file.startswith("epochs:"):
            epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
            lstm_dim = int(re.findall('[0-9]+', file.split('-')[-3])[0])
            lstm_layers = int(re.findall('[0-9]+', file.split('-')[-2])[0])
            acc = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
            if acc > best_acc:
                best_params = [epochs, lstm_dim, lstm_layers]
                best_acc = acc
                weights = os.path.join(weightdir, file)
                best_weights = weights
    return best_acc, best_params, best_weights

def get_offsets(text):
    offsets = {}
    bidx = 0
    eidx = 0
    for i, token in enumerate(text.split()):
        eidx = bidx + len(token)
        offsets[i] = (bidx, eidx)
        bidx = eidx + 1
    return offsets

def convert_prediction_to_token_ids(pred):
    return list(np.where(np.array(pred) == 1)[0])
