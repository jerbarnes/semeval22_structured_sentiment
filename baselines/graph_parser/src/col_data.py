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

def find_roots(col_sent):
    roots = []
    for token in col_sent.tokens:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if idx == 0:
                    roots.append(token)
    return list(set(roots))

def sort_tokens(tokens):
    sorted_tokens = []
    sorted_idxs = sorted([token.id for token in tokens])
    for idx in sorted_idxs:
        for token in tokens:
            if token.id == idx:
                sorted_tokens.append(token)
    return sorted_tokens

def get_char_offsets(sorted_tokens):
    char_offsets = []
    idxs = []
    current_idxs = []
    current_bidx = None
    current_eidx = None
    for i, token in enumerate(sorted_tokens):
        bidx, eidx = token.char_offsets
        if current_bidx == None:
            current_bidx = bidx
            current_idxs.append(i)
        if current_eidx == None:
            current_eidx = eidx
            if i not in current_idxs:
                current_idxs.append(i)
        elif eidx > current_eidx and bidx == current_eidx + 1:
            current_eidx = eidx
            if i not in current_idxs:
                current_idxs.append(i)
        else:
            char_offsets.append((current_bidx, current_eidx))
            idxs.append(current_idxs)
            current_idxs = [i]
            current_bidx = bidx
            current_eidx = eidx
    char_offsets.append((current_bidx, current_eidx))
    idxs.append(current_idxs)
    return char_offsets, idxs


def gather_expressions(roots, col_sent):
    expression_tokens = []
    expressions = []
    # find all other expression tokens
    for token in col_sent:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if "exp" in label and token not in roots:
                    expression_tokens.append(token)
    # group them by root
    for root in roots:
        exps = [root]
        for token in expression_tokens:
            for idx, label in token.scope:
                if idx == root.id and token not in exps:
                    exps.append(token)
        # sort them by token id
        exp = sort_tokens(exps)
        # get the char_offsets and token ids for each
        char_offset, token_groups = get_char_offsets(exp)
        # convert everything to strings following json sent_graph format
        tokens = []
        char_offsets = []
        for token_group in token_groups:
            token_string = ""
            for i in token_group:
                token_string += exp[i].form + " "
            tokens.append(token_string.strip())
        for bidx, eidx in char_offset:
            char_offsets.append("{0}:{1}".format(bidx, eidx))
        expressions.append([tokens, char_offsets])
    return expressions

def gather_targets(roots, col_sent):
    targets = []
    # find all target roots
    exp_root_idxs = dict([(token.id, {}) for token in roots])
    for token in col_sent:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if idx in exp_root_idxs and "targ" in label:
                    exp_root_idxs[idx][token.id] = [token]
                    for token2 in col_sent:
                        if len(token2.scope) > 0:
                            for idx2, label2 in token2.scope:
                                if idx2 == token.id and token2 not in exp_root_idxs[idx][token.id]:
                                    exp_root_idxs[idx][token.id].append(token2)
    for root_idx, target_group in exp_root_idxs.items():
        root_targets = []
        for target_idx, target_tokens in target_group.items():
            target_tokens = sort_tokens(target_tokens)
            char_offset, token_groups = get_char_offsets(target_tokens)
            # convert everything to strings following json sent_graph format
            tokens = []
            char_offsets = []
            for token_group in token_groups:
                token_string = ""
                for i in token_group:
                    token_string += target_tokens[i].form + " "
                tokens.append(token_string.strip())
            for bidx, eidx in char_offset:
                char_offsets.append("{0}:{1}".format(bidx, eidx))
            root_targets.append([tokens, char_offsets])
        if len(root_targets) > 0:
            targets.append(root_targets)
        else:
            targets.append([[[], []]])
    return targets

def gather_holders(roots, col_sent):
    holders = []
    # find all target roots
    exp_root_idxs = dict([(token.id, {}) for token in roots])
    for token in col_sent:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if idx in exp_root_idxs and "holder" in label:
                    exp_root_idxs[idx][token.id] = [token]
                    for token2 in col_sent:
                        if len(token2.scope) > 0:
                            for idx2, label2 in token2.scope:
                                if idx2 == token.id and token2 not in exp_root_idxs[idx][token.id]:
                                    exp_root_idxs[idx][token.id].append(token2)
    for root_idx, holder_group in exp_root_idxs.items():
        root_holders = []
        for holder_idx, holder_tokens in holder_group.items():
            holder_tokens = sort_tokens(holder_tokens)
            char_offset, token_groups = get_char_offsets(holder_tokens)
            # convert everything to strings following json sent_graph format
            tokens = []
            char_offsets = []
            for token_group in token_groups:
                token_string = ""
                for i in token_group:
                    token_string += holder_tokens[i].form + " "
                tokens.append(token_string.strip())
            for bidx, eidx in char_offset:
                char_offsets.append("{0}:{1}".format(bidx, eidx))
            root_holders.append([tokens, char_offsets])
        if len(root_holders) > 0:
            holders.append(root_holders)
        else:
            holders.append([[[], []]])
    return holders

def get_polarities(roots):
    polarities = []
    for root in roots:
        polarity = None
        for idx, label in root.scope:
            if "exp" in label:
                polarity = label.split("-")[1]
        polarities.append(polarity)
    return polarities

def convert_col_sent_to_json(col_sent):
    sent_json = {
                 "sent_id": col_sent.id,
                 "text": col_sent.text,
                 "opinions": []
                }
    # assign character offsets to each token
    i = 0
    for token in col_sent.tokens:
        j = i + len(token.form)
        token.char_offsets = (i, j)
        assert col_sent.text[i:j] == token.form
        i = j + 1

    # find all roots, i.e. 0:exp-(Positive|Neutral|Negative)
    roots = find_roots(col_sent)

    # gather any other tokens belonging to sentiment expressions
    expressions = gather_expressions(roots, col_sent)

    # get polarities
    polarities = get_polarities(roots)

    # get targets corresponding to sentiment expression
    targets = gather_targets(roots, col_sent)

    # get holders corresponding to sentiment expression
    holders = gather_holders(roots, col_sent)

    assert len(expressions) == len(polarities) == len(targets) == len(holders)

    # put these into opinion dictionaries
    for i, root in enumerate(roots):
        """
        opinion = {"Source": [[], []],
                   "Target": [[], []],
                   "Polar_expression": [[], []],
                   "Polarity": "",
                   "Intensity": "average"
                   }
        """
        for targ in targets[i]:
            for holder in holders[i]:
                opinion = {"Source": holder,
                           "Target": targ,
                           "Polar_expression": expressions[i],
                           "Polarity": polarities[i],
                           "Intensity": "Standard"
                           }
                sent_json["opinions"].append(opinion)
    return sent_json

def convert_conllu_to_json(conllu_sents):
    return [convert_col_sent_to_json(sent) for sent in conllu_sents]
    #for i, sent in enumerate(conllu_sents):
    #    try:
    #        convert_col_sent_to_json(sent)
    #    except:
    #        print(i)

if __name__ == "__main__":
    import json

    sentences = list(read_col_data("../sentiment_graphs/darmstadt_unis/head_final/test.conllu"))

    json_sentences = convert_conllu_to_json(sentences)

    col_sent = sentences[5]
    col_sent2 = sentences[22]

    with open("predictions.json", "w") as outfile:
        json.dump(json_sentences, outfile)
