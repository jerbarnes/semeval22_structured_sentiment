from collections import Counter

UNK = "<UNK>"
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"

class Vocab:
    def __init__(self, unk=UNK, pad=PAD, bos=BOS, eos=EOS):
        self.w2i = {}
        self.i2w = {}

        self.unk = unk
        self.add(unk)
        self.pad = pad
        self.add(pad)
        self.bos = bos
        self.add(bos)
        self.eos = eos
        self.add(eos)

    def __len__(self):
        return len(self.w2i)

    def add(self, token):
        """
        returns False if token is new, True otherwise
        updates self.w2i and i2w with new id
        """
        if token not in self.w2i:
            self.w2i[token] = len(self.w2i)
            self.i2w[self.w2i[token]] = token
            return False
        return True

    def get(self, token):
        return self.w2i.get(token, self.w2i[self.unk])

        
# first make counts to then threshold...

def make_vocabs(sentences, threshold=0):
    forms   = Vocab()
    norms   = Vocab()
    lemmas  = Vocab()
    uposs   = Vocab()
    xposs   = Vocab()
    synrels = Vocab()
    semrels = Vocab()
    chars   = Vocab()
    scoperels = Vocab()

    cnt_forms   = Counter()
    cnt_norms   = Counter()
    cnt_lemmas  = Counter()
    cnt_uposs   = Counter()
    cnt_xposs   = Counter()
    cnt_synrels = Counter()
    cnt_semrels = Counter()
    cnt_chars   = Counter()
    cnt_scoperels = Counter()
    for sen in sentences:
        for token in sen:
            cnt_forms[token.form]     += 1
            cnt_norms[token.norm]     += 1
            cnt_lemmas[token.lemma]   += 1
            cnt_uposs[token.upos]     += 1
            cnt_xposs[token.xpos]     += 1
            cnt_synrels[token.deprel] += 1
            for _, l in token.deps:
                cnt_semrels[l] += 1
            for _, l in token.scope:
                cnt_scoperels[l] += 1
            for char in token.form:
                cnt_chars[char] += 1
    
    def add_entries(vocab, cnt, threshold=threshold):
        for k,v in filter(lambda x: x[1] >= threshold, cnt.items()):
            vocab.add(k)
        return True

    add_entries(forms, cnt_forms, 7)
    add_entries(norms, cnt_norms, 7)
    add_entries(lemmas, cnt_lemmas, 7)
    add_entries(uposs, cnt_uposs)
    add_entries(xposs, cnt_xposs)
    add_entries(synrels, cnt_synrels)
    add_entries(semrels, cnt_semrels)
    add_entries(chars, cnt_chars)
    add_entries(scoperels, cnt_scoperels)

    # Vocabs for semrels might need "_" as input (or add in parser)

    return forms, norms, lemmas, uposs, xposs, synrels, semrels, chars, scoperels

class Vocabs:
    def __init__(self, forms, norms, lemmas, uposs, xposs, synrels, semrels, chars, scoperels):
        self.forms   = forms
        self.norms   = norms
        self.lemmas  = lemmas
        self.uposs   = uposs
        self.xposs   = xposs
        self.synrels = synrels
        self.semrels = semrels
        self.chars   = chars
        self.scoperels = scoperels
        #self.vocabs = [self.forms, self.norms, self.lemmas, self.uposs, self.xposs, self.synrels, self.semrels, self.chars, self.scoperels]
        # only used to match the different targets
        # therefore 3*scope [cue, scope, scope-]
        self.rels = [None, self.synrels, self.semrels, self.scoperels, self.scoperels, self.scoperels]

    def __iter__(self):
        for voc in self.rels:
            yield voc

    def __len__(self):
        return len(self.rels)
    
    def __getitem__(self, index):
        return self.rels[index]

    def __setitem__(self, index, value):
        self.rels[index] = value

if __name__ == "__main__":
    import sys
    import col_data as cd
    import pickle
    sentences = []
    for fn in sys.argv[2:]:
        sentences.extend(cd.read_col_data(fn))
    #sentences = cd.read_col_data(sys.argv[1])
    forms, norms, lemmas, uposs, xposs, synrels, semrels, chars, scoperels = make_vocabs(sentences)
    print([len(v.w2i) for v in [forms, norms, lemmas, uposs, xposs, synrels, semrels, chars, scoperels]])
    vocabs = Vocabs(forms, norms, lemmas, uposs, xposs, synrels, semrels, chars, scoperels)
    #print(synrels.w2i, semrels.w2i, scoperels.w2i)
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(vocabs, fh)
