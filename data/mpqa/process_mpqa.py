from mpqa_datahelpers import collect_opinion_entities
import numpy as np
import os
import json
import stanza
from tqdm import tqdm

from nltk.tokenize.simple import SpaceTokenizer

tk = SpaceTokenizer()


class Subelement():
    def __init__(self, type, texts, offsets):
        self.type = type
        self.texts = texts
        self.offsets = offsets

    def to_dict(self):
        offset_text = ["{}:{}".format(*offset) for offset in self.offsets]
        return [self.texts, offset_text]

class Opinion():
    def __init__(self, source=None,
                 target=None,
                 polar_expression=None,
                 polarity=None,
                 intensity=None):
        self.source = source
        self.target = target
        self.polar_expression = polar_expression
        self.polarity = polarity
        self.intensity = intensity

    def normalize_polarity(self, polarity):
        if "positive" in polarity.lower():
            return "Positive"
        if "negative" in polarity.lower():
            return "Negative"
        if "neutral" in polarity.lower():
            return "Neutral"
        if "both" in polarity.lower():
            return "Neutral"

    def normalize_intensity(self, intensity):
        if "high" in intensity:
            return "Strong"
        if "extreme" in intensity:
            return "Strong"
        if "low" in intensity:
            return "Weak"
        if "medium" in intensity:
            return "Average"
        if "neutral" in intensity:
            return "Average"
        if "Standard" in intensity:
            return "Average"

    def to_dict(self):
        opinion_dict = {"Source": self.source.to_dict(),
                        "Target": self.target.to_dict(),
                        "Polar_expression": self.polar_expression.to_dict(),
                        "Polarity": self.normalize_polarity(self.polarity),
                        "Intensity": self.normalize_intensity(self.intensity)
                        }
        return opinion_dict

class Sentence():
    def __init__(self, sent_id, text, sentence_offsets, tokens, opinions=None):
        self.sent_id = sent_id
        self.text = text
        self.sentence_offsets = sentence_offsets
        self.tokens = tokens
        if not opinions:
            self.opinions = []
        else:
            self.opinions = opinions

    def update_text(self):
        new_text = []
        bidx = 0
        for token in self.tokens:
            eidx = bidx + len(token.text)
            token.update_new_offset((bidx, eidx))
            new_text.append(token.text)
            bidx = eidx + 1
        self.text = " ".join(new_text).strip()

    def add_opinion(self, opinion):
        self.opinions.append(opinion)

    def remove_opinions(self):
        self.opinions = []

    def update_holder(self):
        sent_bidx, sent_eidx = self.sentence_offsets
        for opinion in self.opinions:
            texts = opinion.source.texts
            offsets = opinion.source.offsets
            new_texts = []
            new_offsets = []
            for (bidx, eidx), text in zip(offsets, texts):
                if (bidx >= sent_bidx) and (eidx <= sent_eidx) and (text not in new_texts):
                    new_texts.append(text)
                    new_offsets.append((bidx, eidx))
            # Just take the first holder, as this is usually the most salient one in the sentnece
            if len(new_texts) > 0:
                text = [new_texts[0]]
                offset = [new_offsets[0]]
            else:
                text, offset = [], []
            opinion.source = Subelement("Source", text, offset)

    def update_opinion_offsets(self):
        new_opinions = []
        for opinion in self.opinions:
            new_opinion = Opinion()
            for subelement in ["Source", "Target", "Polar_expression"]:
                texts = []
                offsets = []
                if subelement == "Source":
                    sub_texts = opinion.source.texts
                    sub_offsets = opinion.source.offsets
                if subelement == "Target":
                    sub_texts = opinion.target.texts
                    sub_offsets = opinion.target.offsets
                if subelement == "Polar_expression":
                    sub_texts = opinion.polar_expression.texts
                    sub_offsets = opinion.polar_expression.offsets
                for text, offset in zip(sub_texts, sub_offsets):
                    in_tokens = []
                    bidx, eidx = offset
                    for token in self.tokens:
                        tbidx, teidx = token.original_offset
                        if (tbidx >= bidx) and (teidx <= eidx):
                            in_tokens.append(token)
                    # If in_tokens is empty, search within the sentence for the text
                    if len(in_tokens) == 0:
                        try:
                            new_bidx = self.text.index(text)
                            new_eidx = new_bidx + len(text)
                            texts.append(self.text[new_bidx:new_eidx])
                            offsets.append((new_bidx, new_eidx))
                        except ValueError:
                            #print('{}:{} not found in "{}":{} with polar expression "{}":{}'.format(text, offset, self.text, self.sentence_offsets, opinion.polar_expression.texts, opinion.polar_expression.offsets))
                            #print("-" * 40)
                            pass
                    else:
                        new_bidx = in_tokens[0].new_offset[0]
                        new_eidx = in_tokens[-1].new_offset[-1]
                        texts.append(" ".join(t.text for t in in_tokens))
                        offsets.append((new_bidx, new_eidx))
                new_subelement = Subelement(subelement, texts, offsets)
                if subelement == "Source":
                    new_opinion.source = new_subelement
                if subelement == "Target":
                    new_opinion.target = new_subelement
                if subelement == "Polar_expression":
                    new_opinion.polar_expression = new_subelement
                    new_opinion.polarity = opinion.polarity
                    new_opinion.intensity = opinion.intensity
            new_opinions.append(new_opinion)
        self.opinions = new_opinions

    def to_dict(self):
        sent_dict = {"sent_id": self.sent_id,
                     "text": self.text,
                     "opinions": [o.to_dict() for o in self.opinions]}
        return sent_dict


class Token():
    def __init__(self, token_id, text, original_offset, new_offset=None):
        self.token_id = token_id
        self.text = text
        self.original_offset = original_offset
        self.new_offset = new_offset

    def update_new_offset(self, offset):
        self.new_offset = offset

def closest_holder(exp_scope, holder_scopes):
    exp_off1 = int(exp_scope.split(":")[0])
    h = np.array([i[0] for i in holder_scopes])
    idx = np.argmin(np.abs(h - exp_off1))
    return holder_scopes[idx]

def get_all_holder_ids(holder):
    holder_ids = holder.split(",")
    for i in range(len(holder_ids)):
        for j in range(len(holder_ids)):
            if i < j:
                h = "{0},{1}".format(holder_ids[i], holder_ids[j])
                holder_ids.append(h)
    return holder_ids

def get_sents(text, fname, nlp):
    sents = []
    tagged = nlp(text)
    for i, sentence in enumerate(tagged.sentences):
        sent_id = fname + "-" + str(i)
        text = sentence.text
        sent_bidx = sentence.tokens[0].start_char
        sent_eidx = sentence.tokens[-1].end_char
        #sent_bidx = int(sentence.tokens[0].misc.split("|")[0].split("=")[1])
        #sent_eidx = int(sentence.tokens[-1].misc.split("|")[1].split("=")[1])
        offsets = (sent_bidx, sent_eidx)
        sent_tokens = []
        for token in sentence.tokens:
            #offset_info = token.misc
            #boff, eoff = offset_info.split("|")
            #bidx = int(boff.split("=")[1])
            #eidx = int(eoff.split("=")[1])
            bidx = token.start_char
            eidx = token.end_char
            sent_tokens.append(Token(token.id, token.text, (bidx, eidx)))

        sents.append(Sentence(sent_id,
                              text,
                              offsets,
                              sent_tokens))
    return sents


def match_opinions_to_sents(sents, opinions):
    for opinion in opinions:
        bidx, eidx = opinion.polar_expression.offsets[0]
        for sent in sents:
            sbidx, seidx = sent.sentence_offsets
            if (bidx >= sbidx) and (eidx <= seidx):
                sent.add_opinion(opinion)


def get_opinions(lre_file, text, agents, attitudes, attitudes_type, targets):
    opinions = []
    for i, line in enumerate(open(lre_file)):
        line = line.strip()
        line_tab = line.split("\t")
        # skip the first few lines
        if i < 5:
            continue
        if line_tab[3] == "GATE_direct-subjective" and len(line_tab) > 4:
            exp_scope = line_tab[1].replace(",", ":")
            off1, off2 = exp_scope.split(":")
            off1 = int(off1)
            off2 = int(off2)
            exp_tokens = text[off1:off2]
            expression = Subelement("Polar_expression",
                                    [exp_tokens],
                                    [(off1, off2)])
            #print("Expression: " + str(expression))

            arguments = line_tab[4]

            # get the polarity
            if len(arguments.split("polarity=")) > 1:
                ds_polarity = arguments.split("polarity=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                #print("Polarity: " + str(ds_polarity))

                # get the intensity
                if len(arguments.split("expression-intensity=")) > 1:
                    expression_intensity = arguments.split("expression-intensity=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not expression_intensity:
                        expression_intensity = "average"
                else:
                    expression_intensity = "average"
                    #print("intensity: " + expression_intensity)

                # get the holder
                if len(arguments.split("nested-source=")) > 1:
                    holder = arguments.split("nested-source=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not holder:
                        holder = Subelement("Source", [], [])
                    else:
                        holder_ids = get_all_holder_ids(holder)
                        holder_scopes = []
                        for holder_id in holder_ids:
                            if len(agents[holder_id]) > 0:
                                holder_scopes.extend(tuple(i) for i in agents[holder_id])
                        #closest = closest_holder(exp_scope, holder_scopes)
                        #holder_tokens = text[closest[0]:closest[1]]
                        holder_tokens = [text[s[0]:s[1]] for s in holder_scopes]
                        if holder_tokens == "":
                            #holder = [[], []]
                            holder = Subelement("Source", [], [])
                        else:
                            holder = Subelement("Source",
                                                holder_tokens,
                                                holder_scopes)
                    #                            [tuple(closest)])
                    #print("holder: " + str(holder))

                # keep only those with sentiment attitudes
                if len(arguments.split("attitude-link=")) > 1:

                    # attitude-link="a4, a6" ---> a4,a6
                    attitude_ids = arguments.split("attitude-link=")[1].split('"')[1].replace(' ', '').split(',')

                    for aid, att_id in enumerate(attitude_ids):
                        att_type = attitudes_type[att_id] if attitudes_type[att_id] else 'none'

                        target_ids = attitudes[att_id]

                        for target_id in target_ids:
                            target_spans = targets[target_id]
                            target_tokens = ""
                            target_offsets = []
                            for span in target_spans:
                                off1 = int(span[0])
                                off2 = int(span[1])
                                target_offsets.append((off1, off2))
                                target_tokens += text[off1:off2]

                            target = Subelement("Target",
                                                [target_tokens],
                                                target_offsets)

                        if "sentiment" in att_type and expression.texts != ['']:
                            opinion = Opinion(source=holder,
                                              target=target,
                                              polar_expression=expression,
                                              polarity=ds_polarity,
                                              intensity=expression_intensity)
                            opinions.append(opinion)
    return opinions

def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):
    """
    char_offsets: list of str
    token_offsets: list of tuples

    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19"]
    >>> token_offsets =
    [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

    >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
    >>> (2,3,4)
    """
    token_idxs = []
    #
    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        intoken = False
        for i, (b, e) in enumerate(token_offsets):
            if b == bidx:
                intoken = True
            if intoken:
                token_idxs.append(i)
            if e == eidx:
                intoken = False
    return frozenset(token_idxs)

def process_file(fname, nlp):
    #print(fname)
    lre_file = "database.mpqa.2.0/man_anns/{0}/gateman.mpqa.lre.2.0".format(fname)
    doc_file = "database.mpqa.2.0/docs/{0}".format(fname)
    text = open(doc_file).read()
    sents = get_sents(text, fname, nlp)
    agents, attitudes, attitudes_type, targets = collect_opinion_entities(lre_file)

    opinions = get_opinions(lre_file, text, agents, attitudes, attitudes_type, targets)

    match_opinions_to_sents(sents, opinions)

    processed_sents = []
    for sent in sents:
        sent.update_text()
        sent.update_holder()
        sent.update_opinion_offsets()
        sentence = sent.to_dict()
        # remove any annotations with incorrect polar expressions
        new_opinions = []
        for opinion in sentence["opinions"]:
            offset_text, offset = opinion["Polar_expression"]
            polarity = opinion["Polarity"]
            text = sentence["text"]
            exp_char_idxs = [i.split(":") for i in offset]

            token_offsets = list(tk.span_tokenize(text))
            exp = convert_char_offsets_to_token_idxs(offset, token_offsets)
            if offset != [] and exp != frozenset() and polarity is not None:
                new_opinions.append(opinion)
            else:
                print(sentence["sent_id"])
        sentence["opinions"] = new_opinions
        processed_sents.append(sentence)
    return processed_sents


def main():
    train = [l.strip() for l in open("datasplit/filelist_train0").readlines()]
    dev = [l.strip() for l in open("datasplit/filelist_dev").readlines()]
    test = [l.strip() for l in open("datasplit/filelist_test0").readlines()]
    data = [("train", train), ("dev", dev), ("test", test)]

    nlp = stanza.Pipeline("en", processors='tokenize')

    for name, fnames in data:

        processed = []

        for fname in tqdm(fnames):
            new = process_file(fname, nlp)
            processed.extend(new)

        if name in ["test"]:
            for sent in processed:
                sent["opinions"] = []

        with open(os.path.join("{0}.json".format(name)), "w") as out:
            json.dump(processed, out)


if __name__ == "__main__":
    main()
