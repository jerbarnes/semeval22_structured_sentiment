import torch
import torch.nn as nn
from enhanced_lstm import EnhancedLSTM
from utils import create_parameter
from awd.locked_dropout import LockedDropout


class AbstractCharModel(torch.nn.Module):
    """Generates a word model using a single layer unidirectional LSTM"""

    def __init__(self, lstm_type, char_vocab, char_emb_size, word_emb_size,
                 hidden_size, ff_dropout, recurrent_dropout,
                 dropout_char_linear, emb_dropout_type):
        super().__init__()
        self.char_lstm = EnhancedLSTM(
            lstm_type,
            char_emb_size,
            hidden_size,
            num_layers=1,
            ff_dropout=ff_dropout,
            recurrent_dropout=recurrent_dropout,
            bidirectional=False)
        self.char_embedding = nn.Embedding(len(char_vocab), char_emb_size)
        self.char_transform = torch.nn.Linear(hidden_size, word_emb_size)
        self.dropout_char_linear = dropout_char_linear
        self.locked_dropout = LockedDropout()

        if emb_dropout_type == "replace":
            self.drop_token = create_parameter(1, word_emb_size)
        elif emb_dropout_type == "zero":
            self.drop_token = torch.zeros(1, word_emb_size)
        else:
            raise "Unsupported embedding dropout type"

    @staticmethod
    def char_model_factory(model_type, *args, **kwargs):
        if model_type == "convolved":
            return ConvolvedCharModel(*args, **kwargs)
        elif model_type == "single":
            return SingleCharModel(*args, **kwargs)
        else:
            raise NotImplementedError(
                "{model_type} is not a valid attention type".format(model_type))

    def embed_chars(self, batch_vocab):
        # Embed all words in batch vocabulary -> (words x chars x emb)
        return self.char_embedding(batch_vocab)

    def forward(self, char_embeddings, chars, dropout_embedding=0):
        """Takes a batch vocabulary and an index mapping from batched words to
        that vocabulary. Runs the vocabulary through a unidirectional LSTM 
        and uses the hidden state of the last word as the character based word embedding.
        Returns the index mapping expressed in the new char-based word-embeddings"""
        # TODO: should use last cell state instead?

        output = self.char_lstm(char_embeddings, None, chars.voc_lengths)
        last_indices = chars.voc_lengths - 1

        # (batch_vocab_length x hidden)
        word_embeddings = output[torch.arange(chars.vocab_size), last_indices]

        # (batch x seq x hidden)
        embedded_words = word_embeddings[chars.index_mapping]
        embedded_words = self.locked_dropout(
            embedded_words, batch_first=True, p=self.dropout_char_linear)

        # (batch x seq x hidden) -> (batch x seq x emb_size)
        embedded_words = self.char_transform(embedded_words)

        if self.training and dropout_embedding > 0:
            unique_words, inverse_mapping = chars.index_mapping.unique(
                return_inverse=True)
            unique_words.bernoulli_(dropout_embedding)
            #dropped_entires = unique_words[inverse_mapping].to(torch.uint8)
            dropped_entires = unique_words[inverse_mapping].to(torch.bool)

            if dropped_entires.any():
                embedded_words[dropped_entires] = self.drop_token

        return embedded_words


class SingleCharModel(AbstractCharModel):
    """Generates a word model using a single layer unidirectional LSTM 
    taking a character embedding per time step"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, chars, dropout_embedding):
        # Embed all words in batch vocabulary -> (words x chars x emb)
        char_embeddings = super().embed_chars(chars.batch_vocab)

        return super().forward(char_embeddings, chars, dropout_embedding)


class ConvolvedCharModel(AbstractCharModel):
    """Generates a word model using a single layer unidirectional LSTM 
    convolving over three character embeddings at each time step"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Convolve over 3 char embs at the time with SAME padding
        self.conv = nn.Conv1d(
            in_channels=kwargs["char_emb_size"],
            out_channels=kwargs["char_emb_size"],
            kernel_size=3,
            padding=1)

    def forward(self, chars, dropout_embedding):
        # Embed all words in batch vocabulary -> (words x chars x emb)
        char_embeddings = super().embed_chars(chars.batch_vocab)

        # (words x chars x emb) -> (words x emb x chars)
        char_embeddings.transpose_(1, 2)
        # Still (words x emb x chars)
        char_embeddings = self.conv(char_embeddings)
        # (words x emb x chars) -> (words x chars x emb)
        char_embeddings.transpose_(1, 2)

        return super().forward(char_embeddings, chars, dropout_embedding)
