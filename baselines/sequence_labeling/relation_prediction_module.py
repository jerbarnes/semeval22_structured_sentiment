import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
import torchtext
import string
import os
from nltk.tokenize.simple import SpaceTokenizer
from convert_to_bio import get_bio_holder, get_bio_target, get_bio_expression, replace_with_labels
import itertools
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from WordVecs import WordVecs
import numpy as np
import pickle
import argparse

from utils import Vocab, prepare_sequence, RelationDataset, RelationSplit


class Relation_Model(nn.Module):
    def __init__(self, word2idx,
                 embedding_dim,
                 hidden_dim,
                 embedding_matrix=None,
                 pooling="max",
                 lstm_dropout=0.2,
                 word_dropout=0.4,
                 train_embeddings=False):
        super(Relation_Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2idx)
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.pooling = pooling

        if embedding_matrix is not None:
            weight = torch.FloatTensor(embedding_matrix)
            self.word_embeds = nn.Embedding.from_pretrained(weight, freeze=False)
            self.word_embeds.requires_grad = train_embeddings
        else:
            self.word_embeds = nn.Embedding(len(word2idx), embedding_dim)
        self.criterion = nn.BCELoss()

        self.e1_embeds = nn.Embedding(2, embedding_dim)
        self.e2_embeds = nn.Embedding(2, embedding_dim)

        self.sent_lstm = nn.LSTM(embedding_dim,
                                 hidden_dim,
                                 num_layers=1,
                                 bidirectional=True)

        self.e1_lstm = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=1,
                               bidirectional=True)

        self.e2_lstm = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=1,
                               bidirectional=True)


        # Maps the output of the LSTM into tag space.
        self.sigmoid = nn.Sigmoid()
        self.word_dropout = nn.Dropout(word_dropout)
        self.ff = nn.Linear(hidden_dim * 6, 1)

    def init_hidden1(self, batch_size=1):
        h0 = torch.zeros((self.sent_lstm.num_layers*(1+self.sent_lstm.bidirectional),
                                  batch_size, self.sent_lstm.hidden_size))
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def forward(self, sent, e1, e2, batch_sizes):
        batch_size = batch_sizes[0]
        emb = self.word_embeds(sent.data)
        emb = self.word_dropout(emb)
        packed_emb = PackedSequence(emb, batch_sizes)
        self.hidden = self.init_hidden1(batch_size)
        output, (hn, cn) = self.sent_lstm(packed_emb, self.hidden)
        #text_rep = hn.reshape(batch_size, self.hidden_dim * 2)
        o, _ = pad_packed_sequence(output, batch_first=True)
        if self.pooling == "max":
            text_rep, _ = o.max(dim=1)
        else:
            text_rep = o.mean(dim=1)

        emb = self.e1_embeds(e1.data)
        packed_emb = PackedSequence(emb, batch_sizes)
        self.hidden = self.init_hidden1(batch_size)
        output, (hn, cn) = self.e1_lstm(packed_emb, self.hidden)
        #e1_rep = hn.reshape(batch_size, self.hidden_dim * 2)
        o, _ = pad_packed_sequence(output, batch_first=True)
        if self.pooling == "max":
            e1_rep, _ = o.max(dim=1)
        else:
            e1_rep = o.mean(dim=1)

        emb = self.e2_embeds(e2.data)
        packed_emb = PackedSequence(emb, batch_sizes)
        self.hidden = self.init_hidden1(batch_size)
        output, (hn, cn) = self.e2_lstm(packed_emb, self.hidden)
        #e2_rep = hn.reshape(batch_size, self.hidden_dim * 2)
        o, _ = pad_packed_sequence(output, batch_first=True)
        if self.pooling == "max":
            e2_rep, _ = o.max(dim=1)
        else:
            e2_rep = o.mean(dim=1)

        conc = torch.cat((text_rep, e1_rep, e2_rep), dim=1)

        pred = self.ff(conc)

        return self.sigmoid(pred)

    def predict(self, dataloader):
        preds = []
        self.eval()
        for sent_id, sent, e1, e2, label in tqdm(dataloader):
            batch_sizes = sent.batch_sizes
            pred = self.forward(sent, e1, e2, batch_sizes)
            preds.extend(pred)
        pred_labels = [1 if i > .5 else 0 for i in preds]
        self.train()
        return pred_labels, preds

    def fit(self, train_loader, dev_loader,  epochs=10):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=args.LEARNING_RATE)
        best_dev_f1 = 0.0

        for epoch in range(epochs):
            self.train()
            batch_loss = 0
            batches = 0
            preds, golds = [], []
            pbar = tqdm(train_loader, desc="")
            for sent_id, sent, e1, e2, label in pbar:
                self.zero_grad()
                batch_sizes = sent.batch_sizes
                pred = self.forward(sent, e1, e2, batch_sizes)
                preds.extend(pred)
                golds.extend(label)
                loss = self.criterion(pred, label.float())
                pbar.set_description("Epoch: {0} Loss: {1:.3f}".format(epoch + 1, loss))
                batch_loss += loss.data
                batches += 1
                loss.backward()
                optimizer.step()
            pred_labels = [1 if i > .5 else 0 for i in preds]
            golds = [int(i) for i in golds]
            f1 = f1_score(pred_labels, golds, average="macro")

            print("Train Loss: {0:.3f}".format(batch_loss / batches))
            print("Train f1: {0:.3f}".format(f1))

            f1, loss = self.test_model(dev_loader)

            print("Dev f1: {0:.3f}".format(f1))

            if f1 > best_dev_f1:
                    best_dev_f1 = f1
                    print("NEW BEST DEV F1: {0:.3f}".format(f1))


                    basedir = os.path.join("saved_models",
                                           "relation_prediction",
                                           "{0}".format(args.DATADIR))
                    outname = "epochs:{0}-lstm_dim:{1}-lstm_layers:{2}-lr:{3}-pooling:{4}-devf1:{5:.3f}".format(epoch + 1, self.sent_lstm.hidden_size, self.sent_lstm.num_layers, args.LEARNING_RATE, args.POOLING, f1)
                    modelfile = os.path.join(basedir,
                                             outname)
                    os.makedirs(basedir, exist_ok=True)
                    print("saving model to {0}".format(modelfile))
                    if not args.save_all:
                        for file in os.listdir(basedir):
                            if file.startswith("epochs:"):
                                os.remove(os.path.join(basedir, file))
                    torch.save(self.state_dict(), modelfile)

    def test_model(self, dataloader):
        preds, golds = [], []
        self.eval()
        batch_loss = 0
        batches = 0
        for sent_id, sent, e1, e2, label in tqdm(dataloader):
            batches += 1
            batch_sizes = sent.batch_sizes
            pred = self.forward(sent, e1, e2, batch_sizes)
            preds.extend(pred)
            golds.extend(label)
            loss = self.criterion(pred, label.float())
            batch_loss += loss.data
        pred_labels = [1 if i > .5 else 0 for i in preds]
        golds = [int(i) for i in golds]
        f1 = f1_score(pred_labels, golds, average="macro")
        print("F1: {0:.3f}".format(f1))
        full_loss = batch_loss / batches
        self.train()
        return f1, full_loss


def distance(sent, e1, e2):
    e1_idxs = [sent.index(e1[0]), sent.index(e1[-1])]
    e2_idxs = [sent.index(e2[0]), sent.index(e2[-1])]
    m = abs(max(e1_idxs) - min(e2_idxs))
    n = abs(max(e2_idxs) - min(e1_idxs))
    return min((m, n))

def find_best_rel_model(saved_models_dir):
    best_dev_f1 = 0.0
    best_model = ""
    for file in os.listdir(saved_models_dir):
        if file.startswith("epochs:"):
            f1 = float(file.split(":")[-1])
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_model = os.path.join(saved_models_dir, file)
    return best_dev_f1, best_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_true")
    parser.add_argument("--EMBEDDINGS", "-emb", default="../../../embeddings/blse/google.txt")
    parser.add_argument("--DATADIR", "-data", default="opener_en")
    parser.add_argument("--DEVDATA", action="store_true")
    parser.add_argument("--OUTDIR", "-odir", default="saved_models")
    parser.add_argument("--POOLING", default="max", help="max or mean (default is max)")
    parser.add_argument("--LEARNING_RATE", default=0.001, type=float)
    parser.add_argument("--save_all", action="store_true", help="if true, saves all models, otherwise, saves only best model")

    args = parser.parse_args()
    print(args)

    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    print("loading embeddings from {0}".format(args.EMBEDDINGS))
    embeddings = WordVecs(args.EMBEDDINGS)
    emb_dim = embeddings.vector_size
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by one to avoid overwriting the UNK token
    # at index 0
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 1
    vocab.update(with_unk)

    # Import datasets
    # This will update vocab with words not found in embeddings
    dataset = RelationDataset(vocab, True)
    train = dataset.get_split(os.path.join("data",
                                           "relations",
                                           args.DATADIR,
                                           "train.json"))

    vocab.set_vocab()

    if args.DEVDATA:
        dev = dataset.get_split(os.path.join("data",
                                             "relations",
                                             args.DATADIR,
                                             "dev.json"))
    else:
        # split train into train and dev
        split_idx = int(len(train) * .8)
        dev = RelationSplit(train[split_idx:])
        train = RelationSplit(train[:split_idx])

    # Get new embedding matrix so that words not included in pretrained embeddings have a random embedding

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, emb_dim))
    new_embeddings = np.zeros((diff, emb_dim))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))

    # create relation prediction model
    rel_model = Relation_Model(vocab,
                               embedding_dim=emb_dim,
                               hidden_dim=args.HIDDEN_DIM,
                               embedding_matrix=new_matrix,
                               pooling=args.POOLING)


    train_loader = DataLoader(train,
                              batch_size=20,
                              collate_fn=train.collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(dev,
                            batch_size=20,
                            collate_fn=train.collate_fn,
                            shuffle=False)

    # Save the model parameters
    param_file = (dict(vocab.items()),
                  new_matrix.shape,
                  args.HIDDEN_DIM,
                  args.NUM_LAYERS,
                  args.POOLING)

    basedir = os.path.join("saved_models",
                           "relation_prediction",
                           "{0}".format(args.DATADIR))
    outfile = os.path.join(basedir,
                           "params.pkl")
    print("Saving model parameters to " + outfile)
    os.makedirs(basedir, exist_ok=True)

    with open(outfile, "wb") as out:
        pickle.dump(param_file, out)


    print("training...")
    rel_model.fit(train_loader, dev_loader, epochs=10)

    print("loading best model...")
    f1, best_model = find_best_rel_model(os.path.join("saved_models",
                                                  "relation_prediction",
                                                  "{0}".format(args.DATADIR)))
    rel_model.load_state_dict(torch.load(best_model))

    vocab.update_idx2w()

    rel_model.test_model(dev_loader)
