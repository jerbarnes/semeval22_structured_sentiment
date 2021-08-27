import numpy as np
import pickle
from scipy.spatial.distance import cosine
import gensim

class WordVecs(object):
    """Import word2vec files saved in txt format.
    Creates an embedding matrix and two dictionaries
    (1) a word to index dictionary which returns the index
    in the embedding matrix
    (2) a index to word dictionary which returns the word
    given an index.
    """


    def __init__(self, file, vocab=None, encoding='utf8'):
        self.vocab = vocab
        self.encoding = encoding
        (self.vocab_length, self.vector_size, self._matrix,
         self._w2idx, self._idx2w) = self._read_vecs(file)

    def __getitem__(self, y):
        try:
            return self._matrix[self._w2idx[y]]
        except KeyError:
            raise KeyError
        except IndexError:
            raise IndexError
        

    def _read_vecs(self, file):
        """Assumes that the first line of the file is
        the vocabulary length and vector dimension."""

        if file.endswith(".txt"):
            model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=False, unicode_errors="replace")
        elif file.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(file, "r") as archive:
                stream = archive.open("model.bin")
                model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True, unicode_errors="replace")
        vocab_length, vec_dim = model.vectors.shape
        emb_matrix = model.vectors
        w2idx = dict([(w, i.index) for w, i in model.vocab.items()])
        idx2w = dict([(i.index, w) for w, i in model.vocab.items()])

        return vocab_length, vec_dim, emb_matrix, w2idx, idx2w
