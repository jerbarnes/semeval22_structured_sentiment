# sentence id: # sid 12345
# one token per line
# empty lines between sentences
# file ends with an empty line
import re
import numpy as np

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

def pair(x):
    a,*b = x.split(":")
    b = ":".join(b)
    #a,b = x.split(":")[0:2]
    #a = a.split(".")[0]
    return int(a), b

class Sentence:
    def __init__(self, id, tokens, tokens_full, text):
        # tokens_full contains multi words 4-5 nella in la
        self.id = id
        self.tokens = tokens
        self.tokens_full = tokens_full
        self.text = text

    def print_text(self):
        return self.text

    def __repr__(self):
        return "\n".join([f"# sent_id = {self.id}"] + [f"# text = {self.text}"] + [str(t) for k,t in sorted(self.tokens_full.items(), key=lambda x: x[0])] + [""])

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        return self.tokens[index]

    def __setitem__(self, index, value):
        self.tokens[index] = value
    
    def make_matrix(self, sss, label=False, w2i=None):
        """sss has to be either syn sem or scope"""
        n = len(self.tokens) + 1
        matrix = np.zeros((n,n))
        try:
            for t in self:
                m = t.id
                if sss == "syn":
                    h = t.head
                    l = t.deprel
                    if h == "_": continue
                    matrix[h,m] = w2i[l] if label else 1
                elif sss == "sem":
                    for h,l in t.deps:
                        matrix[h,m] = w2i[l] if label else 1
                elif sss == "scope":
                    for h,l in t.scope:
                        matrix[h,m] = w2i[l] if label else 1
                elif sss == "cues":
                    for h,l in t.scope:
                        if l in "cue mwc".split():
                            matrix[h,m] = w2i[l] if label else 1
                elif sss == "scope-":
                    for h,l in t.scope:
                        if l in "scope event".split():
                            matrix[h,m] = w2i[l] if label else 1
        except KeyError:
            # sneaky back-off to unlabelled
            try:
                return self.make_matrix(sss,False,None)
            except IndexError:
                # and another back-off if there are no heads
                # pass and return zero-matrix
                pass
        return matrix

    def update_parse(self, matrix, sss, i2w=None):
        """
        update each Token by matrix which are labelled matrices
        if no i2w is given, unlabelled parses are applied
        """
        for token in self.tokens:
            if sss == "scope" or sss == "scope-":
                token.scope = []
            elif sss == "syn":
                token.head = -1
                #token.head = 1 
                token.deprel = "_"
            elif sss == "sem":
                token.deps = []
            #elif sss == "syn+sem":
            #    self.update_parse(matrix[0], "syn", i2w[0])
            #    self.update_parse(matrix[1], "sem", i2w[1])
            #    return True
        for h in range(len(matrix)):
            for m in range(1, len(matrix)):
                if matrix[h,m] > 0:
                    if sss == "scope" or sss == "scope-":
                        if i2w is None:
                            l = "_"
                        else:
                            l = i2w[matrix[h,m]]
                        self[m-1].scope.append((h,l))
                        self[m-1].print_scope = True
                    elif sss == "syn":
                        self[m-1].head = h
                        if i2w is not None:
                            self[m-1].deprel = i2w[matrix[h,m]]
                    elif sss == "sem":
                        if i2w is None:
                            l = "_"
                        else:
                            l = i2w[matrix[h,m]]
                        self[m-1].deps.append((h,l))
        return True


                
        
class Token:
    def __init__(self, id, form, lemma, upos, xpos, feats,
            head, deprel, deps, misc, scope=None):
        self.id = int(id)
        self.form = form
        self.norm = normalize(form)
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        try:
            self.head = int(head)
        except ValueError:
            self.head = "_"
        self.deprel = deprel
        if deps != "_":
            self.deps = [pair(x) for x in deps.split("|")]
        else:
            self.deps = []
        self.misc = misc
        if scope is not None:
            self.print_scope = True
            if scope != "_": # ids of cues separated by |
                #self.scope = [int(i) for i in scope.split("|")]
                self.scope = [pair(x) for x in scope.split("|")]
            else:
                self.scope = []
        else:
            self.scope = []


    def __repr__(self):
        strlist = [str(self.id), self.form, self.lemma, self.upos, self.xpos,
                    self.feats, str(self.head), self.deprel]
        if self.deps != []:
            strlist.append("|".join(["{}:{}".format(i,l) for i,l in self.deps]))
        else:
            strlist.append("_")
            #strlist.append(":_")
        strlist.append(self.misc)
        if self.print_scope:
            if self.scope != []:
                #strlist.append("|".join([str(i) for i in self.scope]))
                strlist.append("|".join(["{}:{}".format(i,l) for i,l in self.scope]))
            else:
                strlist.append("_")

        return "\t".join(strlist)

class TokenFaux(Token):
    def __init__(self, id, form, lemma, upos, xpos, feats,
            head, deprel, deps, misc, scope="_"):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = "_"
        self.deprel = deprel
        self.deps = []
        self.misc = misc


def read_col_data(fname):
    """
    yields Sentences
    """
    tokens = []
    tokens_full = {}
    sid = -1
    text = ""
    with open(fname) as fhandle:
        for line in fhandle:
            if line.startswith("# sent_id"):
                sid = line.split("=")[1].strip()
            elif line.startswith("# text"):
                text = line.split("=")[1].strip()
            elif line.startswith("#sid"):
                sid = line.split()[1].strip()
            elif line.startswith("#"):
                continue
            elif line == "\n":
                yield Sentence(sid, tokens, tokens_full, text)
                tokens = []
                tokens_full = {}
            else:
                try:
                    tokens.append(Token(*line.strip().split("\t")))
                    tokens_full[len(tokens)] = tokens[-1]
                except ValueError:
                    tokens_full[len(tokens)+ 0.5] = (TokenFaux(*line.strip().split("\t")))
                except TypeError:
                    print(line)


