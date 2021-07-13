import torch
import torch.nn as nn

from attention import Attention, DotProductAttention
from utils import dropout_mask
from enhanced_lstm import EnhancedLSTM
from char_model import AbstractCharModel
from awd.locked_dropout import LockedDropout


class BaseLSTM(nn.Module):
    """A LSTM based network predicting labeled and unalabeled dependencies between words"""

    def __init__(self, vocabs, external, settings):
        super().__init__()
        self.use_external = not settings.disable_external
        self.use_lemma = not settings.disable_lemma
        self.use_pos = not settings.disable_pos
        self.use_form = not settings.disable_form
        self.use_char = not settings.disable_char
        self.unfactorized = settings.unfactorized
        self.emb_dropout_type = settings.emb_dropout_type

        self.use_elmo = settings.use_elmo

        self.init_embeddings(vocabs, external, settings)

        if self.use_char:
            self.char_model = AbstractCharModel.char_model_factory(
                model_type=settings.char_implementation,
                lstm_type=settings.lstm_implementation,
                char_vocab=vocabs.chars,
                char_emb_size=settings.dim_char_embedding,
                word_emb_size=settings.dim_embedding,
                hidden_size=settings.hidden_char_lstm,
                ff_dropout=settings.dropout_char_ff,
                recurrent_dropout=settings.dropout_recurrent_char,
                dropout_char_linear=settings.dropout_char_linear,
                emb_dropout_type=settings.emb_dropout_type,
                )
        main_lstm_input = (
            settings.dim_embedding *
            (self.use_form + self.use_pos + self.use_lemma + self.use_char) +
            (external.dim * self.use_external + self.use_elmo * 100))
        self.main_lstm_input = main_lstm_input
        self.lstm = EnhancedLSTM(
            settings.lstm_implementation,
            main_lstm_input,
            settings.hidden_lstm,
            num_layers=settings.layers_lstm,
            ff_dropout=settings.dropout_main_ff,
            recurrent_dropout=settings.dropout_main_recurrent,
            bidirectional=True)


        self.dim_lstm = settings.hidden_lstm
        self.dim_embedding = settings.dim_embedding

        self.dropout_embedding = settings.dropout_embedding
        self.dropout_label = settings.dropout_label
        self.dropout_main_recurrent = settings.dropout_main_recurrent
        self.dropout_main_ff = settings.dropout_main_ff
        self.locked_dropout = LockedDropout()


    def init_embeddings(self, vocabs, external, settings):
        # Let the last embedding entry represent <DROP> tokens
        extra = self.emb_dropout_type == "replace"

        if not settings.disable_form:
            self.word_embedding = nn.Embedding(
                len(vocabs.norms) + extra, settings.dim_embedding)

        if not settings.disable_pos:
            self.pos_embedding = nn.Embedding(
                len(vocabs.xposs) + extra, settings.dim_embedding)

        if not settings.disable_lemma:
            self.lemma_embedding = nn.Embedding(
                len(vocabs.lemmas) + extra, settings.dim_embedding)

        # Glove is frozen -> no need for drop token
        if not settings.disable_external:
            self.external_embedding = nn.Embedding.from_pretrained(external.data)

    def get_embeddings(self, indices, embedding, drop=True):
        if not self.training or self.dropout_embedding == 0 or not drop:
            return embedding(indices)

        # Currently drops the same embedding type for all sequences in the batch

        if self.emb_dropout_type == "replace":
            unique_indices, inverse_indices = indices.unique(
                return_inverse=True)
            unique_indices[torch.rand(unique_indices.size()) < self.
                           dropout_embedding] = -1
            return embedding.weight[unique_indices[inverse_indices]]
        elif self.emb_dropout_type == "zero":
            size = (embedding.weight.size(0), 1)
            mask = dropout_mask(embedding.weight.data, size,
                                self.dropout_embedding)

            masked_embedding = embedding.weight * mask

            return masked_embedding[indices]
        else:
            raise "Unsupported embedding dropout type"

    def merge_features(self, word_indices, pos_indices, external_indices,
                       lemma_indices, char_features, elmo_vecs):
        features = []

        if self.use_form:
            features.append(
                self.get_embeddings(word_indices, self.word_embedding))

        if self.use_pos:
            features.append(
                self.get_embeddings(pos_indices, self.pos_embedding))

        if self.use_external:
            features.append(
                self.get_embeddings(
                    external_indices, self.external_embedding, drop=False))

        if self.use_lemma:
            features.append(
                self.get_embeddings(lemma_indices, self.lemma_embedding))

        if self.use_char:
            features.append(char_features)

        if self.use_elmo: # if use_elmo
            features.append(elmo_vecs)

        return torch.cat(features, 2)


    def forward(self, seq_lengths, chars, word_indices,
                pos_indices, external_indices, lemma_indices, elmo_vecs=None):
        word_indices = word_indices
        pos_indices = pos_indices
        external_indices = external_indices if self.use_external else None
        lemma_indices = lemma_indices if self.use_lemma else None
        seq_lengths = seq_lengths

        char_features = self.char_model(
            chars, self.dropout_embedding) if self.use_char else None

        merged_features = self.merge_features(word_indices, pos_indices,
                                              external_indices, lemma_indices,
                                              char_features, elmo_vecs)

        # (batch x seq x 2*dim_lstm)
        output = self.lstm(merged_features, None, seq_lengths)
        return output, merged_features



class SecondLSTM(nn.Module):
    """A LSTM based network predicting labeled and unalabeled dependencies between words"""

    def __init__(self, settings):
        super().__init__()


        fnn_input = settings.hidden_lstm * 2
        self.lstm_to_fnn = nn.Linear(fnn_input, settings.hidden_lstm)

        main_lstm_input =  settings.hidden_lstm

        self.lstm = EnhancedLSTM(
            settings.lstm_implementation,
            main_lstm_input,
            settings.hidden_lstm,
            num_layers=settings.layers_lstm,
            ff_dropout=settings.dropout_main_ff,
            recurrent_dropout=settings.dropout_main_recurrent,
            bidirectional=True)


        self.dim_lstm = settings.hidden_lstm
        self.dim_embedding = settings.dim_embedding

        self.dropout_embedding = settings.dropout_embedding
        self.dropout_label = settings.dropout_label
        self.dropout_main_recurrent = settings.dropout_main_recurrent
        self.dropout_main_ff = settings.dropout_main_ff
        self.locked_dropout = LockedDropout()


    def forward(self, seq_lengths, inputs):
        seq_lengths = seq_lengths

        inputs = self.lstm_to_fnn(inputs)
        # (batch x seq x 2*dim_lstm)
        output = self.lstm(inputs, None, seq_lengths)

        return output

class Scorer(nn.Module):

    def __init__(self, n_labels, settings, unfactorized, lonely_only=True):
        super().__init__()
        self.unfactorized = unfactorized


        if lonely_only:
            fnn_input = settings.hidden_lstm * 2
        else:
            fnn_input = settings.hidden_lstm * 2 * 2

        self.label_head_fnn = nn.Linear(fnn_input, settings.dim_mlp)
        self.label_dep_fnn = nn.Linear(fnn_input, settings.dim_mlp)


        self.label_attention = Attention.label_factory(
            settings.dim_mlp,
            n_labels, settings.attention)

        self.dim_lstm = settings.hidden_lstm

        self.dropout_label = settings.dropout_label
        self.dropout_main_ff = settings.dropout_main_ff
        self.locked_dropout = LockedDropout()

        if not self.unfactorized:
            self.edge_head_fnn = nn.Linear(fnn_input, settings.dim_mlp)
            self.edge_dep_fnn = nn.Linear(fnn_input, settings.dim_mlp)

            self.edge_attention = Attention.edge_factory(
                settings.dim_mlp,
                settings.attention)

            self.dropout_edge = settings.dropout_edge



    def get_scores(self, lstm_outputs, dropout, head_fnn,
                   dep_fnn, attention):
        # head, dep: [batch x sequence x mlp]

        lstm_outputs = self.locked_dropout(
            lstm_outputs, batch_first=True, p=dropout)

        head = head_fnn(lstm_outputs)
        dep = dep_fnn(lstm_outputs)


        head = self.locked_dropout(head, batch_first=True, p=dropout)
        dep = self.locked_dropout(dep, batch_first=True, p=dropout)

        # (batch x seq x seq)
        return attention(head, dep)

    def get_edge_scores(self, lstm_outputs):
        return self.get_scores(
            lstm_outputs,
            dropout=self.dropout_edge,
            head_fnn=self.edge_head_fnn,
            dep_fnn=self.edge_dep_fnn,
            attention=self.edge_attention)

    def get_label_scores(self, lstm_outputs):
        return self.get_scores(
            lstm_outputs,
            dropout=self.dropout_label,
            head_fnn=self.label_head_fnn,
            dep_fnn=self.label_dep_fnn,
            attention=self.label_attention)

    def forward(self, lstm_outputs, attended_outputs=None):

        if attended_outputs is not None:
            lstm_outputs = torch.cat((lstm_outputs, attended_outputs), dim=2)

        edge_scores = self.get_edge_scores(
            lstm_outputs) if not self.unfactorized else None
        label_scores = self.get_label_scores(lstm_outputs)

        return edge_scores, label_scores

class BiLSTMModel(nn.Module):
    """A LSTM based network predicting labeled and unalabeled dependencies between words"""

    def __init__(self, vocabs, external, settings):
        super().__init__()

        only_lonely = True
        self.settings = settings

        self.n_labels_other = 0 if vocabs[settings.ot] is None else len(vocabs[settings.ot])
        self.n_labels = len(vocabs[settings.pt])

        self.other_scorer = None
        self.bridge = None
        self.combine = None
        self.helpers = settings.helpers
        self.base = BaseLSTM(vocabs, external, settings)
        if settings.use_elmo:
            self.scalelmo = nn.Linear(settings.vec_dim, 100)
        else:
            self.scalelmo = None
        if settings.ot:
            self.other_scorer = Scorer(self.n_labels_other, settings, False, True)

        if settings.ot or settings.helpers:
            only_lonely = False
            if settings.bridge == "dpa":
                self.bridge = DotProductAttention(settings.dim_mlp)
            elif settings.bridge == "dpa+":
                self.combine = nn.Linear(settings.hidden_lstm*4, settings.hidden_lstm*2)
                def bridge(x,y):
                    a = DotProductAttention(settings.dim_mlp)(x,y)
                    b = DotProductAttention(settings.dim_mlp)(x.transpose(-2,-1),y)
                    c = self.combine(torch.cat((a,b), -1))
                    return c
                self.bridge = bridge
            elif settings.bridge == "gcn":
                self.bridge = GCN(settings.hidden_lstm*2, int(settings.hidden_lstm / 2), settings.hidden_lstm*2, settings.gcn_layers, settings, self.n_labels_other)
            elif settings.bridge == "simple":
                self.bridge = lambda x,y: x.transpose(-2, -1).float() @ y

        self.scorer = Scorer(self.n_labels, settings, settings.unfactorized, only_lonely)
        print(self.n_labels)

        #self.gcn = GCN(self.base.main_lstm_input, int(settings.hidden_lstm / 2), settings.hidden_lstm*2, settings.gcn_layers, settings, self.n_labels_other)


    def forward(self, other_targets, seq_lengths, chars, word_indices,
                pos_indices, external_indices, lemma_indices, elmo_vecs=None):


        if self.scalelmo:
            elmo_scaled = self.scalelmo(elmo_vecs)
        else:
            elmo_scaled = None
        # (batch x seq x 2*dim_lstm)
        if torch.cuda.is_available():
            print("model start")
            print(torch.cuda.memory_allocated(self.settings.device)/10**6)
            print(torch.cuda.memory_cached(self.settings.device)/10**6)
            torch.cuda.empty_cache()
            print(torch.cuda.memory_cached(self.settings.device)/10**6)
        output, inputs = self.base( seq_lengths, chars, word_indices, pos_indices,
                external_indices, lemma_indices, elmo_scaled)

        if torch.cuda.is_available():
            print("post bilstm")
            print(torch.cuda.memory_allocated(self.settings.device)/10**6)
            print(torch.cuda.memory_cached(self.settings.device)/10**6)
            torch.cuda.empty_cache()
            print(torch.cuda.memory_cached(self.settings.device)/10**6)

        dp_output = None
        other_edge_scores = None
        other_label_scores = None
        if self.helpers:
            h = self.helpers[0]
            dp_output = self.bridge(other_targets[h], output)
            for h in self.helpers[1:]:
                dp_output += self.bridge(other_targets[h], output)
        if self.other_scorer:
            other_edge_scores, other_label_scores = self.other_scorer(output)
            dp_output = self.bridge(other_edge_scores, output)

        if torch.cuda.is_available():
            print("post other")
            print(torch.cuda.memory_allocated(self.settings.device)/10**6)
            print(torch.cuda.memory_cached(self.settings.device)/10**6)
            torch.cuda.empty_cache()
            print(torch.cuda.memory_cached(self.settings.device)/10**6)

        edge_scores, label_scores = self.scorer(output, dp_output)

        return other_edge_scores, other_label_scores, edge_scores, label_scores


class GCN(nn.Module):
    """GCN: adjacency matrix adj, labels, dimensions, layers"""

    def __init__(self, gcn_input, hidden_dim, out_dim, layers, settings, n_labels):
        # TODO from input a reduction so that not everything is bilstm-out
        super().__init__()
        #self.device = device
        self.layers = layers
        self.hidden_dim = hidden_dim


        self.W_parents  = [nn.Linear(gcn_input, self.hidden_dim)]#.to(self.device)]
        self.W_children = [nn.Linear(gcn_input, self.hidden_dim)]#.to(self.device)]
        self.W_self     = [nn.Linear(gcn_input, self.hidden_dim)]#.to(self.device)]
        for l in range(1, layers):
            self.W_parents.append(nn.Linear(self.hidden_dim, self.hidden_dim))#.to(self.device))
            self.W_children.append(nn.Linear(self.hidden_dim, self.hidden_dim))#.to(self.device))
            self.W_self.append(nn.Linear(self.hidden_dim, self.hidden_dim))#.to(self.device))
        self.W_parents = nn.ModuleList(self.W_parents)
        self.W_children = nn.ModuleList(self.W_children)
        self.W_self = nn.ModuleList(self.W_self)
        #self.W_parents  = [nn.Linear(gcn_input, gcn_input).to(self.device) for l in range(layers)]
        #self.W_children = [nn.Linear(gcn_input, gcn_input).to(self.device) for l in range(layers)]
        #self.W_self     = [nn.Linear(gcn_input, gcn_input).to(self.device) for l in range(layers)]

        #self.label_bias = nn.Embedding(n_labels, self.hidden_dim, padding_idx=0)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.last = nn.Linear(self.hidden_dim, out_dim)#.to(self.device)

    def forward(self, adj, inputs, label_scores=None):
        # adj_labeled als Tensor? jede gelabelte Kante ist der jewelige bias vector
        # man muss nen lookup machen?
        #
        #L = self.label_bias(torch.argmax(label_scores, dim=1))
        X = inputs
        #adj = adj.to(self.device)
        #adj = (adj >= 0).float()
        b,n,_ = adj.shape
        #L_ = torch.cat(
        #        [torch.cat([sum(adj[m,i,j] * L[m,i,j,:]
        #                    for j in range(n)) for i in range(n)]
        #                  ).reshape(n,self.hidden_dim)
        #                    for m in range(b)
        #        ]).reshape(b,n,self.hidden_dim)

        # eye = torch.eye(*tuple(adj.shape)[-2:]).reshape((1,
        #     *tuple(adj.shape)[-2:])).repeat((tuple(adj.shape)[0], 1,
        #         1)).to(self.device)


        adj = adj.float()
        #print(torch.argmax(label_scores, dim=1))
        #L_ = self.label_bias((adj * torch.argmax(label_scores, dim=1)).long())
        #Lp = torch.sum(L_, dim=1)
        #Lc = torch.sum(L_, dim=2)
        for l in range(self.layers):
            # sigmoid does not help so far
            #parent_sum = self.W_parents[l](adj.transpose(-2,-1) @ X)
            parent_sum = adj.transpose(-2,-1) @ self.W_parents[l](X)
            #parent_sum = adj.transpose(-2,-1) @ (self.W_parents[l](X) + Lp)
            #parent_sum = adj.transpose(-2,-1) @ self.sigmoid(self.W_parents[l](X))
            #child_sum  = self.W_children[l](adj @ X)
            child_sum  = adj @ self.W_children[l](X)
            #child_sum  = adj @ (self.W_children[l](X) + Lc)
            #child_sum  = adj @ self.sigmoid(self.W_children[l](X))
            #self_sum = self.W_self[l](torch.eye(tuple(adj.shape)) @ X)
            #self_sum = eye @ self.W_self[l](X)
            self_sum = (self.W_self[l](X))
            #self_sum = eye @ self.sigmoid(self.W_self[l](X))

            X = self.relu(parent_sum + child_sum + self_sum)

        return self.last(X)


