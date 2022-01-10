from lxml import etree
from lxml.etree import fromstring
import os
import json
import re

parser = etree.XMLParser(recover=True, encoding='utf8')

def get_sent_spans(text, file):
    tokens = text.split()
    token_spans = []
    for line in open(file):
        if "span" in line and "SentenceOpinionAnalysisResult" not in line:
            try:
                span = re.findall('span="([a-z0-9\._]*)"', line)[0]
                bidx, eidx = span.split("..")
                bidx = int(bidx.split("_")[1])
                bidx -= 1
                eidx = int(eidx.split("_")[1])
                token_spans.append((bidx, eidx))
            except ValueError:
                pass
                #print(line)
                #print(span)
            except IndexError:
                pass
    token_spans.sort()
    #
    sents = [" ".join(tokens[a:b]) for a, b in token_spans]
    doc_spans = []
    for sent in sents:
        bidx = text.index(sent)
        eidx = text.index(sent) + len(sent)
        doc_spans.append((bidx, eidx))
    #
    return doc_spans

def expand_span(span):
    if "," in span:
        spans = span.split(",")
        new_span = []
        for sp in spans:
            if ".." in sp:
                off1, off2 = sp.split("..")
                off1 = int(off1.split("_")[-1])
                off2 = int(off2.split("_")[-1])
                r = list(range(off1, off2+1))
                new_span.extend(["word_" + str(i) for i in r])
            else:
                new_span.extend([sp])
        return new_span
    #
    elif ".." in span:
        off1, off2 = span.split("..")
        off1 = int(off1.split("_")[-1])
        off2 = int(off2.split("_")[-1])
        r = list(range(off1, off2+1))
        span = ["word_" + str(i) for i in r]
    else:
        span = [span]
    return span


def get_sents(sent_file):
    mark_xml = open(sent_file).read().encode('utf8')
    mark_root = fromstring(mark_xml, parser)
    #
    sents = []
    #
    for i in mark_root:
        sent_span = i.get("span")
        sent_span = expand_span(sent_span)
        sents.append(sent_span)
    return sents


def get_opinions(base_file, markable_file):

    polarity_flip_dict = {"positive": "negative",
                          "negative": "positive",
                          "neutral": "negative"}

    increase_strength_dict = {"average": "strong",
                              "weak": "average",
                              "strong": "strong"}

    decrease_strength_dict = {"average": "weak",
                              "weak": "weak",
                              "strong": "average"}

    new = {}
    new["sent_id"] = base_file.split("/")[-1][:-10]
    new["bdir"] = re.findall("DarmstadtServiceReviewCorpus/(.*)/basedata", base_file)[0]


    base_xml = open(base_file).read().encode('utf8')
    mark_xml = open(markable_file).read().encode('utf8')

    base_root = fromstring(base_xml, parser)
    mark_root = fromstring(mark_xml, parser)

    tokens = {}
    spans = {}
    markups = {}

    text = ""
    span_idx = 0

    for i in base_root:
        idx = i.get("id")
        token = i.text
        tokens[idx] = token
        text += token + " "
        begin_span = span_idx
        end_span = span_idx + len(token)
        spans[idx] = (begin_span, end_span)
        span_idx += len(token) + 1

    for i in mark_root:
        idx = i.get("id")
        markups[idx] = i

    opinions = []

    for m in markups.values():
        if m.get("annotation_type") == "opinionexpression":
            idx = m.get("id")
            #print(idx)
            hspan = m.get("opinionholder")
            exp_span = m.get("span")
            tspan = m.get("opiniontarget")
            label = m.get("polarity")
            modifier = m.get("opinionmodifier")
            intensity = m.get("strength")

            # Collect opion holder: text and spans
            if hspan == "empty":
                holder = [[], []]
            elif hspan is None:
                holder = [[], []]
            elif ";" in hspan:
                #print(hspan) # only one example of multiple holders
                hspan = hspan.split(";")[0]
                holder_span = markups[hspan].get("span")
                holder_span = expand_span(holder_span)
                holder_tokens = " ".join([tokens[i] for i in holder_span])
                hld_off1 = spans[holder_span[0]][0]
                hld_off2 = spans[holder_span[-1]][1]
                #hld_off1 = text.find(holder_tokens)
                #hld_off2 = hld_off1 + len(holder_tokens)
                holder = [[holder_tokens], ["{0}:{1}".format(hld_off1, hld_off2)]]
            else:
                holder_span = markups[hspan].get("span")
                holder_span = expand_span(holder_span)
                holder_tokens = " ".join([tokens[i] for i in holder_span])
                hld_off1 = spans[holder_span[0]][0]
                hld_off2 = spans[holder_span[-1]][1]
                #hld_off1 = text.find(holder_tokens)
                #hld_off2 = hld_off1 + len(holder_tokens)
                holder = [[holder_tokens], ["{0}:{1}".format(hld_off1, hld_off2)]]

            # deal with any modified expressions
            # these may change the polarity (negation), intensity (increase)
            # additionally, the offsets for the expressions will need to be updated
            if modifier != "empty" and modifier is not None:
                if ";" in modifier:
                    mod_tokens = ""
                    mod_offs = ""
                    modifiers = modifier.split(";")
                    for modifier in modifiers:
                        modifier = markups[modifier]
                        change = modifier.get("modifier")
                        modifier_span = modifier.get("span")
                        modifier_span = expand_span(modifier_span)
                        mod_toks = " ".join([tokens[i] for i in modifier_span])
                        mod_off1 = spans[modifier_span[0]][0]
                        mod_off2 = spans[modifier_span[-1]][1]
                        #mod_off1 = text.find(mod_tokens)
                        #mod_off2 = mod_off1 + len(mod_tokens)

                        mod_tokens += mod_toks + ";"
                        offs = "{0}:{1}".format(mod_off1, mod_off2)
                        mod_offs += offs + ";"

                        if change == "negation":
                            label = polarity_flip_dict[label]
                        elif change == "increase":
                            intensity = increase_strength_dict[intensity]
                        elif change == "decrease":
                            intensity = decrease_strength_dict[intensity]
                        else:
                            pass
                            #print(change)

                    # remove trailing semicolons
                    mod_offs = mod_offs[:-1]
                    mod_tokens = mod_tokens[:-1]

                else:
                    modifier = markups[modifier]
                    change = modifier.get("modifier")
                    modifier_span = modifier.get("span")
                    modifier_span = expand_span(modifier_span)
                    mod_tokens = " ".join([tokens[i] for i in modifier_span])
                    mod_off1 = spans[modifier_span[0]][0]
                    mod_off2 = spans[modifier_span[-1]][1]
                    #mod_off1 = text.find(mod_tokens)
                    #mod_off2 = mod_off1 + len(mod_tokens)
                    mod_offs = "{0}:{1}".format(mod_off1, mod_off2)

                    if change == "negation":
                        label = polarity_flip_dict[label]
                        #print(change)
                        #print(new_polarity)
                    elif change == "increase":
                        intensity = increase_strength_dict[intensity]
                        #print(new_strength)
                    elif change == "decrease":
                        intensity = decrease_strength_dict[intensity]
                    else:
                        pass
                        #print(change)

            # Collect opinion expression: text, span, polarity, and intensity
            exp_span = expand_span(exp_span)
            #print(exp_span)
            exp_tokens = " ".join([tokens[i] for i in exp_span])

            exp_off1 = spans[exp_span[0]][0]
            exp_off2 = spans[exp_span[-1]][1]
            #exp_off1 = text.find(exp_tokens)
            #exp_off2 = exp_off1 + len(exp_tokens)

            if modifier != "empty" and modifier is not None:
                expression = [[mod_tokens, exp_tokens], ["{0}".format(mod_offs), "{0}:{1}".format(exp_off1, exp_off2)]]
            else:
                expression = [[exp_tokens], ["{0}:{1}".format(exp_off1, exp_off2)]]
            #

            # Collect opinion target: text and spans
            if tspan == "empty":
                target = [[], []]
            elif tspan is None:
                target = [[], []]
            elif ";" in tspan:
                tspans = tspan.split(";")
                for tsp in tspans:
                    target_span = markups[tsp].get("span")
                    target_span = expand_span(target_span)
                    target_tokens = " ".join([tokens[i] for i in target_span])
                    trg_off1 = spans[target_span[0]][0]
                    trg_off2 = spans[target_span[-1]][1]
                    #trg_off1 = text.find(target_tokens)
                    #trg_off2 = trg_off1 + len(target_tokens)
                    target = [[target_tokens], ["{0}:{1}".format(trg_off1, trg_off2)]]

                    # for each target, add an opinion to the list
                    opinions.append({"Source": holder,
                                     "Target": target,
                                     "Polar_expression": expression,
                                     "Polarity": label.title(),
                                     "Intensity": intensity.title()})

            else:
                target_span = markups[tspan].get("span")
                target_span = expand_span(target_span)
                target_tokens = " ".join([tokens[i] for i in target_span])

                trg_off1 = spans[target_span[0]][0]
                trg_off2 = spans[target_span[-1]][1]
                #trg_off1 = text.find(target_tokens)
                #trg_off2 = trg_off1 + len(target_tokens)
                target = [[target_tokens], ["{0}:{1}".format(trg_off1, trg_off2)]]

                opinions.append({"Source": holder,
                                 "Target": target,
                                 "Polar_expression": expression,
                                 "Polarity": label.title(),
                                 "Intensity": intensity.title()})

        # elif m.get("annotation_type") == "polar_target":
        #     idx = m.get("id")
        #     #print(idx)
        #     tspan = m.get("span")
        #     label = m.get("polar_target_polarity")

        #     target_span = expand_span(tspan)
        #     target_tokens = " ".join([tokens[i] for i in target_span])
        #     trg_off1 = spans[target_span[0]][0]
        #     trg_off2 = spans[target_span[-1]][1]
        #     #trg_off1 = text.find(target_tokens)
        #     #trg_off2 = trg_off1 + len(target_tokens)
        #     target = [[target_tokens], ["{0}:{1}".format(trg_off1, trg_off2)]]

        #     # for each target, add an opinion to the list
        #     opinions.append({"Source": [[], []],
        #                      "Target": target,
        #                      "Polar_expression": [[], []],
        #                      "Polarity": label,
        #                      "Intensity": "average"})

    #
    new["text"] = text
    new["opinions"] = opinions
    #
    return new


def get_anns_in_sent(b_sent_idx, e_sent_idx, opinions):
    in_sent = []
    for o in opinions:
        _, exp_idxs = o["Target"]
        for e in exp_idxs:
            bidx, eidx = e.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            if b_sent_idx <= bidx < e_sent_idx:
                in_sent.append(o)
    return in_sent


def get_sentence_level_anns(document_level_anns):

    polarity_dictionary = {None: "neutral",
                           "not_set": "neutral"}

    sent_anns = []
    bdir = document_level_anns["bdir"]
    idx = document_level_anns["sent_id"]
    text = document_level_anns["text"]
    opinions = document_level_anns["opinions"]

    # Get sentence and word character offsets
    sents_data = "DarmstadtServiceReviewCorpus/{0}/markables/{1}_SentenceOpinionAnalysisResult_level.xml".format(bdir, idx)
    sent_spans = get_sent_spans(text, sents_data)

    # Move from document to sentence level annotations
    for i, sent_span in enumerate(sent_spans):
        sent_idx = idx + "-" + str(i + 1)

        # get the sent bidx and eidx for sent
        b_sent_idx = sent_span[0]
        e_sent_idx = sent_span[-1]

        # get the text for the sentence
        sent_text = text[b_sent_idx:e_sent_idx]

        # find the annotations that are in this sentence
        in_sent = get_anns_in_sent(b_sent_idx,
                                   e_sent_idx,
                                   opinions)

        #print(sent_text)
        #print(in_sent)

        new_opinions = []

        # change their offsets to the sentence level
        for op in in_sent:
            try:
                holder = op["Source"]
                new_texts = []
                new_idxs = []
                for i, idxs in enumerate(holder[1]):
                    bidx, eidx = idxs.split(":")
                    new_bidx = int(bidx) - b_sent_idx
                    new_eidx = int(eidx) - b_sent_idx
                    if new_bidx >= 0 and new_eidx <= len(sent_text):
                        new_idxs.append("{0}:{1}".format(new_bidx, new_eidx))
                        new_texts.append(holder[0][i])
                    else:
                        try:
                            holder_text = " ".join(holder[0])
                            new_bidx = sent_text.index(holder_text)
                            new_eidx = new_bidx + len(holder_text)
                            new_idxs.append("{0}:{1}".format(new_bidx, new_eidx))
                            new_texts.append(holder[0][i])
                        except ValueError:
                            pass
                holder[0] = new_texts
                holder[1] = new_idxs
            except:
                holder = [[], []]

            try:
                target = op["Target"]
                new_texts = []
                new_idxs = []
                for i, idxs in enumerate(target[1]):
                    bidx, eidx = idxs.split(":")
                    new_bidx = int(bidx) - b_sent_idx
                    new_eidx = int(eidx) - b_sent_idx
                    if new_bidx >= 0 and new_eidx <= len(sent_text):
                        new_idxs.append("{0}:{1}".format(new_bidx, new_eidx))
                        new_texts.append(target[0][i])
                    else:
                        try:
                            tgt_text = " ".join(target[0])
                            new_bidx = sent_text.index(tgt_text)
                            new_eidx = new_bidx + len(tgt_text)
                            new_idxs.append("{0}:{1}".format(new_bidx, new_eidx))
                            new_texts.append(target[0][i])
                        except ValueError:
                            pass
                target[0] = new_texts
                target[1] = new_idxs
            except:
                target = [[], []]

            try:
                expression = op["Polar_expression"]
                new_idxs = []
                for idxs in expression[1]:
                    bidx, eidx = idxs.split(":")
                    new_bidx = int(bidx) - b_sent_idx
                    new_eidx = int(eidx) - b_sent_idx
                    if new_bidx >= 0 and new_eidx <= len(sent_text):
                        new_idxs.append("{0}:{1}".format(new_bidx, new_eidx))
                expression[1] = new_idxs
            except:
                expression = [[], []]

            if op["Polarity"] in polarity_dictionary:
                polarity = polarity_dictionary[op["Polarity"]]
            else:
                polarity = op["Polarity"]
            intensity = op["Intensity"]

            # Create new opinion at the sentence level
            opinion = {"Source": holder,
                       "Target": target,
                       "Polar_expression": expression,
                       "Polarity": polarity.title(),
                       "Intensity": intensity.title()}

            new_opinions.append(opinion)

        sent_new = {"sent_id": sent_idx,
                    "text": sent_text,
                    "opinions": new_opinions
                   }

        sent_anns.append(sent_new)
    return sent_anns

def get_files(current_dir):
    base_files = [i for i in os.listdir(os.path.join(current_dir, "basedata")) if "xml" in i and "~" not in i]
    mark_files = [i.split("_words.xml")[0] + "_OpinionExpression_level.xml"
                  for i in base_files]
    return list(zip(base_files, mark_files))


if __name__ == "__main__":

    basedir = "DarmstadtServiceReviewCorpus"
    processed = {"train": [], "dev": [], "test": []}
    corpus = "universities"

    with open("full_splits.json") as infile:
        splits = json.load(infile)

    current_dir = os.path.join(basedir, corpus)
    ff = get_files(current_dir)

    #break into train, dev splits
    train = [filenames for filenames in ff if filenames[0].split("_words")[0] in splits["train"]]
    dev = [filenames for filenames in ff if filenames[0].split("_words")[0] in splits["dev"]]
    test = [filenames for filenames in ff if filenames[0].split("_words")[0] in splits["test"]]


    # import list of
    problematic_sentences = [line.strip() for line in open("problematic.txt")]

    for subname, subcorpus in [("train", train), ("dev", dev), ("test", test)]:

        for bf, mf in subcorpus:
            bfile = os.path.join(current_dir, "basedata", bf)
            mfile = os.path.join(current_dir, "markables", mf)

            o = get_opinions(bfile, mfile)
            sentence_anns = get_sentence_level_anns(o)
            if subname in ["test"]:
                for s in sentence_anns:
                    s["opinions"] = []

            # remove sentences which have polar expressions with no or incorrect offsets
            for sentence in sentence_anns:
                if sentence["sent_id"] in problematic_sentences:
                    new_opinions = []
                    for opinion in sentence["opinions"]:
                        offset_text, offset = opinion["Polar_expression"]
                        text = sentence["text"]
                        offsets = [i.split(":") for i in offset]
                        offsets = sorted([(int(i), int(j)) for i, j in offsets])
                        offset_text2 = [text[b:e] for b, e in offsets]
                        if offset != [] and set(offset_text) == set(offset_text2):
                            new_opinions.append(opinion)
                    sentence["opinions"] = new_opinions

            processed[subname].extend(sentence_anns)



    for subname, subcorpus in processed.items():
        with open(os.path.join("{0}.json".format(subname)), "w") as out:
                    json.dump(subcorpus, out)

