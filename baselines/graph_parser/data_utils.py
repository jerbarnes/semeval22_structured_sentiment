import argparse
import os
import json

from nltk.tokenize.simple import SpaceTokenizer
tk = SpaceTokenizer()


def get_bio_target(opinion):
    try:
        text, idxs = opinion["Target"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        #
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            target_tokens = t.split()
            label = "targ"
            #
            tags = []
            for i, token in enumerate(target_tokens):
                tags.append(label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = text[0].split()
        label = "targ"
        #
        tags = []
        for i, token in enumerate(target_tokens):
            tags.append(label)
        return [(bidx, tags)]

def get_bio_expression(opinion):
    try:
        text, idxs = opinion["Polar_expression"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        #
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            polarity = opinion["Polarity"]
            target_tokens = t.split()
            label = "exp-{0}".format(polarity)
            #
            tags = []
            for i, token in enumerate(target_tokens):
                tags.append(label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = text[0].split()
        label = "exp-{0}".format(polarity)
        #
        tags = []
        for i, token in enumerate(target_tokens):
            tags.append(label)
        return [(bidx, tags)]

def get_bio_holder(opinion):
    try:
        text, idxs = opinion["Source"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        #
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            target_tokens = t.split()
            label = "holder"
            #
            tags = []
            for i, token in enumerate(target_tokens):
                tags.append(label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = text[0].split()
        label = "holder"
        #
        tags = []
        for i, token in enumerate(target_tokens):
            tags.append(label)
        return [(bidx, tags)]


def replace_with_labels(labels, offsets, bidx, tags):
    try:
        token_idx = offsets.index(bidx)
        for i, tag in enumerate(tags):
            labels[i + token_idx] = tag
        return labels
    except:
        return labels


def create_labels(text, opinion):
    """
    Converts a text (each token separated by a space) and an opinion expression
    into a list of labels for each token in the text.
    """
    offsets = [l[0] for l in tk.span_tokenize(text)]
    #
    labels = ["O"] * len(offsets)
    #
    anns = []
    try:
        anns.extend(get_bio_holder(opinion))
    except:
        pass
    try:
        anns.extend(get_bio_target(opinion))
    except:
        pass
    try:
        anns.extend(get_bio_expression(opinion))
    except:
        pass
    #
    for bidx, tags in anns:
        labels = replace_with_labels(labels, offsets, bidx, tags)
    return labels

def create_sentiment_dict(labels, setup="point_to_root", inside_label=False):
    """
    point_to_root: the final token of the sentiment expression is set as the root and all other labels point to this

    head_first: the first token in the sentiment expression is the root, and the for the holder and target expressions, the first token connects to the root, while the other tokens connect to the first

    head final: the final token in the sentiment expression is the root, and the for the holder and target expressions, the final token connects to the root, while the other tokens connect to the final
    """
    sent_dict = {}
    #
    # associate each label with its token_id
    enum_labels = [(i + 1, l) for i, l in enumerate(labels)]
    #
    if setup in ["point_to_root", "head_final"]:
        enum_labels = list(reversed(enum_labels))
    #
    #for token_id, label in reversed(enum_labels):
    for token_id, label in enum_labels:
        if "exp" in label:
            sent_dict[token_id] = "0:{0}".format(label)
            exp_root_id = token_id
            break
    #
    # point_to_root: point to exp_root_id, regardless of expression type
    if setup == "point_to_root":
        for token_id, label in enum_labels:
            if label == "O":
                sent_dict[token_id] = "_"
            else:
                if token_id not in sent_dict.keys():
                    sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
    # head_first or head_final: first/final point to exp_root, others point inside expression
    else:
        for token_id, label in enum_labels:
            if "targ" in label:
                sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
                targ_root_id = token_id
                break
        #
        for token_id, label in enum_labels:
            if "holder" in label:
                sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
                holder_root_id = token_id
                break
        #
        # set other leafs to point to root
        for token_id, label in enum_labels:
            if label == "O":
                sent_dict[token_id] = "_"
            else:
                if token_id not in sent_dict.keys():
                    if inside_label:
                        label = "IN:" + label
                    if "exp" in label:
                        sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
                    elif "targ" in label:
                        sent_dict[token_id] = "{0}:{1}".format(targ_root_id, label)
                    elif "holder" in label:
                        sent_dict[token_id] = "{0}:{1}".format(holder_root_id, label)
    return sent_dict

def create_conll_sent_dict(conllu_sent):
    conll_dict = {}
    for line in conllu_sent.split("\n"):
        if line != "":
            token_id = int(line.split()[0])
            conll_dict[token_id] = line
    return conll_dict

def combine_labels(token_labels):
    final_label = ""
    for l in token_labels:
        if l == "_":
            pass
        else:
            if final_label == "":
                final_label = l
            else:
                final_label += "|" + l
    if final_label == "":
        return "_"
    return final_label


def combine_sentiment_dicts(sentiment_dicts):
    combined = {}
    for i in sentiment_dicts[0].keys():
        labels = [s[i] for s in sentiment_dicts]
        final_label = combine_labels(labels)
        combined[i] = final_label
    return combined


def create_sentiment_conll(finegrained_sent,
                           norec_sents,
                           setup="point_to_root",
                           inside_label=False,
                           use_dep_edges=False,
                           use_dep_labels=False
                           ):
    sentiment_conll = ""
    #
    sent_id = finegrained_sent["sent_id"]
    text = finegrained_sent["text"]
    opinions = finegrained_sent["opinions"]
    conll = norec_sents[sent_id]
    conll_dict = create_conll_sent_dict(conll)
    t2e = tokenidx2edge(conll)
    t2l = tokenidx2deplabel(conll)
    #
    if len(opinions) > 0:
        labels = [create_labels(text, o) for o in opinions]
    else:
        labels = [create_labels(text, [])]
    #
    sent_labels = [create_sentiment_dict(l,
                                         setup=setup,
                                         inside_label=inside_label) for l in labels]
    if use_dep_edges:
        if use_dep_labels:
            sent_labels = [redefine_root_with_dep_edges(s, t2e, t2l) for s in sent_labels]
        else:
            sent_labels = [redefine_root_with_dep_edges(s, t2e) for s in sent_labels]

    combined_labels = combine_sentiment_dicts(sent_labels)
    #
    for i in conll_dict.keys():
        #print(c[i] + "\t" + sd[i])
        sentiment_conll += conll_dict[i] + "\t" + combined_labels[i] + "\n"
    return sentiment_conll


def redefine_root_with_dep_edges(sent_labels, t2e, t2l=None):
    new_sent_labels = {}
    # If there are no sentiment annotations, return the current labels
    if set(sent_labels.values()) == {'_'}:
        return sent_labels
    # Find the full sentiment expression in the annotation
    exp = []
    exp_label = ""
    for idx, label in sent_labels.items():
        if "exp" in label:
            exp_label = label
            exp.append(idx)
    exp_label = exp_label.split(":")[-1]
    edges = [t2e[i] for i in exp]
    if t2l:
        deplabels = [t2l[i] for i in exp]
    else:
        deplabels = None
    #
    # given the dependency edges in the sentiment expression,
    # find the one that has an incoming edge and set as root
    root = get_const_root(exp, edges, deplabels)
    new_sent_labels[root] = "0:" + exp_label
    #
    # Do the same for the target
    targ = []
    for idx, label in sent_labels.items():
        if "targ" in label:
            targ.append(idx)
    if len(targ) > 0:
        edges = [t2e[i] for i in targ]
        if t2l:
            deplabels = [t2l[i] for i in targ]
        else:
            deplabels = None
        targ_root = get_const_root(targ, edges, deplabels)
        new_sent_labels[targ_root] = "{0}:targ".format(root)
    # Do the same for holder
    holder = []
    for idx, label in sent_labels.items():
        if "holder" in label:
            holder.append(idx)
    if len(holder) > 0:
        edges = [t2e[i] for i in holder]
        if t2l:
            deplabels = [t2l[i] for i in holder]
        else:
            deplabels = None
        holder_root = get_const_root(holder, edges, deplabels)
        new_sent_labels[holder_root] = "{0}:holder".format(root)
    # Now iterate back through the remaining tokens in the sentiment expression
    # and set their edges pointing towards the new root, as well as the target
    # root and holder root
    for idx, label in sent_labels.items():
        if idx not in new_sent_labels:
            if "exp" in label:
                new_sent_labels[idx] = "{0}:IN:{1}".format(root, exp_label)
            elif "targ" in label:
                new_sent_labels[idx] = "{0}:IN:targ".format(targ_root)
            elif "holder" in label:
                new_sent_labels[idx] = "{0}:IN:holder".format(holder_root)
            else:
                new_sent_labels[idx] = label
    return new_sent_labels


def tokenidx2edge(conllu):
    t2e = {}
    for line in conllu.splitlines():
        split = line.split("\t")
        idx = int(split[0])
        edge = int(split[6])
        t2e[idx] = edge
    return t2e

def tokenidx2deplabel(conllu):
    t2e = {}
    for line in conllu.splitlines():
        split = line.split("\t")
        idx = int(split[0])
        edge = split[7]
        t2e[idx] = edge
    return t2e

def get_const_root(token_ids, edges, dep_labels=None):
    # Given token ids, and dependency edges
    # return the token id which has an incoming
    # edge from outside the group
    roots = []
    labels = []
    for i, token in enumerate(token_ids):
        edge = edges[i]
        if edge not in token_ids:
            roots.append(token)
            if dep_labels:
                labels.append(dep_labels[i])
    if len(roots) > 1:
        if dep_labels:
            # If we have the dependency labels, we can use these to decide
            # which token to set as the root
            new_roots = []
            # remove any punctuation and obliques
            for root, dep_label in zip(roots, labels):
                if dep_label != "obl" and dep_label != "punct":
                    new_roots.append(root)
            if len(new_roots) > 0:
                return new_roots[0]
            else:
                return roots[0]
        else:
            # if there's no better way to tell, return the first root
            return roots[0]
    elif len(roots) == 0:
        return token_ids[0]
    else:
        return roots[0]
