import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader

from itertools import chain
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from WordVecs import WordVecs
import argparse
from tqdm import tqdm
import os
import pickle

from utils import Vocab, Label2Idx, prepare_sequence, ExtractionDataset, get_best_run, Split

class Bilstm(nn.Module):

    def __init__(self,
                 word2idx,
                 embedding_matrix,
                 num_labels,
                 tag2idx,
                 embedding_dim,
                 hidden_dim,
                 num_layers=2,
                 lstm_dropout=0.2,
                 word_dropout=0.5,
                 train_embeddings=False):
        super(Bilstm, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2idx)
        self.tag2idx = tag2idx
        self.num_labels = num_labels
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.sentiment_criterion = nn.CrossEntropyLoss()

        weight = torch.FloatTensor(embedding_matrix)
        self.word_embeds = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        self.word_embeds.requires_grad = train_embeddings

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.num_labels)

        # Set up layers for sentiment prediction
        self.word_dropout = nn.Dropout(word_dropout)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)


    def init_hidden1(self, batch_size=1):
        h0 = torch.zeros((self.lstm.num_layers*(1+self.lstm.bidirectional),
                                  batch_size, self.lstm.hidden_size))
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def forward(self, sequences, lengths):
        self.hidden = self.init_hidden1()
        embs = self.word_embeds(sequences)
        packed = pack_padded_sequence(embs, list(lengths), batch_first=True)
        out_packed, (h, c) = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(out_packed, batch_first=True)
        logits = self.hidden2tag(out_unpacked)
        return logits

    def fit(self, train_data, dev_data, epochs=10):

        model_params = list(self.word_embeds.parameters()) + \
                       list(self.lstm.parameters()) +\
                       list(self.hidden2tag.parameters())

        optimizer = torch.optim.Adam(model_params,
                                     lr=0.01)

        #loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss_function = torch.nn.CrossEntropyLoss()

        best_dev_f1 = 0.0
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            pbar = tqdm(train_data, desc="")
            for sent_id, seq, lengths, labels in pbar:
                pred = self.forward(seq, lengths)
                batch_size, max_length, out_dim = pred.shape
                pred = pred.reshape(batch_size * max_length, out_dim)
                labels = labels.view(-1)
                loss = loss_function(pred, labels)
                pbar.set_description("Epoch: {0} Loss: {1:.3f}".format(epoch + 1, loss))
                epoch_loss += loss
                num_batches += 1
                loss.backward()
                optimizer.step()
                model.zero_grad()
            f1 = self.score(dev_data)
            if f1 > best_dev_f1:
                print("New best model: {0:.3f} on dev".format(f1))
                best_dev_f1 = f1
                basedir = os.path.join("saved_models",
                                       "extraction_models",
                                       args.DATADIR,
                                       args.ANNOTATION)
                outname = "epochs:{0}-lstm_dim:{1}-lstm_layers:{2}-dev_f1:{3:.3f}".format(epoch + 1, model.lstm.hidden_size, model.lstm.num_layers, f1)
                modelfile = os.path.join(basedir,
                                         outname)
                os.makedirs(basedir, exist_ok=True)
                print("saving model to {0}".format(modelfile))
                if not args.save_all:
                    for file in os.listdir(basedir):
                        if file.startswith("epochs:"):
                            os.remove(os.path.join(basedir, file))
                torch.save(model.state_dict(), modelfile)
            #print("Loss for epoch {0}: {1}".format(epoch + 1, epoch_loss.data / num_batches))

    def predict(self, test_data, label2idx=None, annotation=None):
        preds = []
        sent_ids = []
        for sent_id, seq, lengths, labels in test_data:
            sent_ids.extend(sent_id)
            pred = self.forward(seq, lengths)
            pred = pred.argmax(2)
            for pr, length in zip(pred, lengths):
                p = pr[:length]
                if label2idx and annotation:
                    p = label2idx.idxs2labels(p.tolist(), annotation=annotation)
                preds.append(p)
        return sent_ids, preds

    def score(self, test_data):
        final_preds = []
        final_labels = []
        for sent_id, seq, lengths, labels in test_data:
            preds = self.forward(seq, lengths)
            preds = preds.argmax(2)
            for pred, length, label in zip(preds, lengths, labels):
                pred = pred[:length]
                label = label[:length]
                final_preds.extend(pred)
                final_labels.extend(label)
        f1 = f1_score(final_labels, final_preds, average="macro")
        print("F1: {0:.3f}".format(f1))
        return f1



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
    parser.add_argument("--ANNOTATION", "-ann", default="expressions", help="which ")
    parser.add_argument("--save_all", action="store_true", help="if true, saves all models, otherwise, saves only best model")

    args = parser.parse_args()
    print(args)

    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    embeddings = WordVecs(args.EMBEDDINGS)
    w2idx = embeddings._w2idx
    embedding_dim = embeddings.vector_size

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by one to avoid overwriting the UNK token
    # at index 0
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 1
    vocab.update(with_unk)

    label2idx = Label2Idx()
    annotation = args.ANNOTATION

    # Import datasets
    # This will update vocab with words not found in embeddings
    extraction_dataset = ExtractionDataset(vocab, label2idx, False)

    source_train = extraction_dataset.get_split(os.path.join("data",
                                                             "extraction",
                                                             args.DATADIR,
                                                             "train.json"),
                                                lower_case=False,
                                                annotation=annotation
                                                )
    vocab.set_vocab()

    if args.DEVDATA:
        source_dev = extraction_dataset.get_split(os.path.join("data",
                                                               "extraction",
                                                               args.DATADIR,
                                                               "dev.json"),
                                                  lower_case=False,
                                                  annotation=annotation
                                                  )
    else:
        # split train into train and dev
        split_idx = int(len(source_train) * .8)
        source_dev = Split(source_train[split_idx:])
        source_train = Split(source_train[:split_idx])

    tag2idx = label2idx.label2idx[annotation]
    num_labels = len(set(tag2idx.values()))

    train_loader = DataLoader(source_train,
                              batch_size=args.BATCH_SIZE,
                              collate_fn=source_train.collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(source_dev,
                            batch_size=args.BATCH_SIZE,
                            collate_fn=source_train.collate_fn,
                            shuffle=False)

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, embedding_dim))
    new_embeddings = np.zeros((diff, embedding_dim))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))

    # Save the model parameters
    param_file = (dict(vocab.items()),
                  new_matrix.shape,
                  num_labels,
                  label2idx,
                  args.HIDDEN_DIM,
                  args.NUM_LAYERS,
                  args.ANNOTATION)

    basedir = os.path.join("saved_models",
                           "extraction_models",
                           args.DATADIR,
                           args.ANNOTATION)
    outfile = os.path.join(basedir,
                           "params.pkl")
    print("Saving model parameters to " + outfile)
    os.makedirs(basedir, exist_ok=True)
    with open(outfile, "wb") as out:
        pickle.dump(param_file, out)

    model = Bilstm(vocab,
                   new_matrix,
                   num_labels,
                   tag2idx,
                   embedding_dim,
                   args.HIDDEN_DIM,
                   args.NUM_LAYERS,
                   train_embeddings=args.TRAIN_EMBEDDINGS)

    model.fit(train_loader, dev_loader, epochs=10)
    f1, params, best_weights = get_best_run(basedir)
    model.load_state_dict(torch.load(best_weights))
    model.eval()

    model.score(dev_loader)
