from torch.utils.data import Dataset
import torch
import vocab as vcb
import col_data as cd
import h5py


class IndexEntry:
    """Convert and store a Sentence in index format"""

    def __init__(self, sentence, vocabs, external, settings, elmo_vecs, vec_dim=1024):

        word_indices = []
        pos_indices = []
        external_indices = []
        lemma_indices = []
        char_indices = []
        elmo_vectors = []
        # init with root
        pos_style = settings.pos_style
        word_indices.append(vocabs.norms.get(vcb.BOS))
        if pos_style == "xpos":
            pos_indices.append(vocabs.xposs.get(vcb.BOS))
        else:
            pos_indices.append(vocabs.uposs.get(vcb.BOS))
        external_indices.append(external.get(vcb.BOS))
        lemma_indices.append(vocabs.lemmas.get(vcb.BOS))
        char_indices.append(
            tuple(vocabs.chars.get(vcb.BOS) for c in range(1)))
        if settings.use_elmo:
            elmo_vectors.append(torch.zeros(vec_dim))
        else:
            elmo_vecs = [None for _ in range(len(sentence))]


        for token, vec in zip(sentence, elmo_vecs):
            word_indices.append(vocabs.norms.get(token.norm))
            if pos_style == "xpos":
                pos_indices.append(vocabs.xposs.get(token.xpos))
            else:
                pos_indices.append(vocabs.uposs.get(token.upos))
            external_indices.append(external.get(token.norm))
            lemma_indices.append(vocabs.lemmas.get(token.lemma))
            char_indices.append(
                tuple(vocabs.chars.get(c) for c in token.form))
            if settings.use_elmo:
                elmo_vectors.append(torch.Tensor(vec))


        self._id = sentence.id
        self.char_indices = char_indices
        self.word_indices = torch.LongTensor(word_indices)
        self.pos_indices = torch.LongTensor(pos_indices)
        self.external_indices = torch.LongTensor(external_indices)
        self.lemma_indices = torch.LongTensor(lemma_indices)
        self.targets = [torch.zeros(len(sentence)+1, len(sentence)+1),
                        sentence.make_matrix("syn", True, vocabs.synrels.w2i),
                        sentence.make_matrix("sem", True, vocabs.semrels.w2i),
                        sentence.make_matrix("cue", True, vocabs.scoperels.w2i),
                        sentence.make_matrix("scope", True, vocabs.scoperels.w2i),
                        sentence.make_matrix("scope-", True, vocabs.scoperels.w2i)]

        if settings.use_elmo:
            self.elmo_vecs = torch.stack(elmo_vectors)
            self.elmo_vecs.requires_grad = False
        else:
            self.elmo_vecs = None

        # no gradients require for external vectors and indices
        self.word_indices.requires_grad = False
        self.pos_indices.requires_grad = False
        self.external_indices.requires_grad = False
        self.lemma_indices.requires_grad = False


class MyDataset(Dataset):
    def __init__(self, data_path, vocabs, external, settings, elmo, vec_dim):
        super().__init__()

        self.external = external
        self.vocabs = vocabs
        self.index_entries = None
        self.settings = settings
        self.vec_dim = vec_dim

        pos_style = settings.pos_style
        target_style=settings.target_style
        other_target_style=settings.other_target_style

        self.use_elmo = settings.use_elmo
        self._load_data(data_path, pos_style, target_style, other_target_style, elmo)


    def _load_data(self, data_path, pos_style, target_style, other_target_style, elmo):
        print("Loading data from {}".format(data_path))
        data = cd.read_col_data(data_path)
        #with h5py.File(elmo, 'r') as f:
        #    for sen in f:
        #        #print(sen)
        #        for word, vec in zip(sen.split("\t"), f[sen]):
        #            print(word, vec)

        if self.use_elmo:
            felmo = h5py.File(elmo, "r")

        self.index_entries = []
        for sentence in data:
            #print(sentence.id)
            #print(len(sentence), len(felmo[sentence.id]))
            if self.use_elmo:
                self.index_entries.append(IndexEntry(
                    sentence, self.vocabs, self.external, self.settings, felmo[sentence.id], self.vec_dim)
                    )
            else:
                self.index_entries.append(IndexEntry(
                    sentence, self.vocabs, self.external, self.settings, None)
                    )

        if self.use_elmo:
            felmo.close()
        print("Done")
        #return data

    def __len__(self):
        return len(self.index_entries)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        entry = self.index_entries[idx]
        targets = [torch.Tensor(target) for target in entry.targets]
        if self.use_elmo:
            return (entry._id, targets, entry.char_indices,
                    entry.word_indices, entry.pos_indices, entry.external_indices,
                    entry.lemma_indices, entry.elmo_vecs)
        return (entry._id, targets, entry.char_indices,
                entry.word_indices, entry.pos_indices, entry.external_indices,
                entry.lemma_indices)



class Glove(vcb.Vocab):
    def __init__(self, fname):
        super().__init__()
        if fname is not None:
            self.read_vectors(fname)
        else:
            self.dim = 0


    def read_vectors(self, fname):
        glove = []
        print("Loading glove vectors")
        with open(fname) as f:
            for line in f:
                line = line.strip().split()
                word = line[0]
                vector = [float(v) for v in line[1:]]
                self.add(word)
                glove.append(vector)
        print("Done")

        self.dim = len(glove[0])

        self.data = torch.tensor(glove, requires_grad=False)



class External(vcb.Vocab):
    def __init__(self, fname):
        super().__init__()
        if fname is not None:
            self.read_vectors(fname)
        else:
            self.dim = 0

    def read_vectors(self, fname):
        iszip = False
        if fname.endswith(".zip"):
            iszip = True
            import zipfile
        import gensim
        print("Loading External Vectors")
        if iszip:
            with zipfile.ZipFile(fname, "r") as archive:
                stream = archive.open("model.bin")
                model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True, unicode_errors="replace")
        else:
            if fname.endswith(".robin"):
                model = gensim.models.KeyedVectors.load(fname)
            else:
                model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True, unicode_errors="replace")

        extra = len(self.w2i)
        self.dim = model.vector_size
        self.data = torch.cat([torch.zeros(extra, self.dim, requires_grad=False), torch.tensor(model.vectors, requires_grad=False)])
        for word in model.index2word:
            self.add(word)
