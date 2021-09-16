from nltk.tokenize.simple import SpaceTokenizer
from convert_to_bio import get_bio_holder, get_bio_target, get_bio_expression, replace_with_labels
import itertools
import json
import os
import numpy as np
import argparse

def create_bio_labels(text, opinion):
    tk = SpaceTokenizer()
    #
    offsets = [l[0] for l in tk.span_tokenize(text)]
    #
    columns = ["Source", "Target", "Polar_expression"]
    labels = {c: ["O"] * len(offsets) for c in columns}
    #
    anns = {c: [] for c in columns}
    # TODO: deal with targets which can have multiple polarities, due to
    # contrasting polar expressions. At present the last polarity wins.
    try:
        anns["Source"].extend(get_bio_holder(opinion))
    except:
        pass
    try:
        anns["Target"].extend(get_bio_target(opinion))
    except:
        pass
    try:
        anns["Polar_expression"].extend(get_bio_expression(opinion))
    except:
        pass
    #
    for c in columns:
        for bidx, tags in anns[c]:
            labels[c] = replace_with_labels(labels[c], offsets, bidx, tags)
    return labels

def convert_to_train_example(sent_id, text, opinions):
    examples = []
    if len(opinions) == 0:
        return []
    elif len(opinions) > 1:
        olabels = []
        ols = []
        for opinion in opinions:
            labels = create_bio_labels(text, opinion)
            ls = {}
            for c in ["Source", "Target", "Polar_expression"]:
                labels[c] = [0 if l == "O" else 1 for l in labels[c]]
                ls[c] = len(set(labels[c]))
            olabels.append(labels)
            ols.append(ls)
            if ls["Target"] > 1:
                examples.append((sent_id, text, labels["Target"], labels["Polar_expression"], 1))
            if ls["Source"] > 1:
                examples.append((sent_id, text, labels["Source"], labels["Polar_expression"], 1))
        # Iterate over possible combinations
        x = range(len(olabels))
        for idx1, idx2 in itertools.combinations(x, 2):
            if ols[idx1]["Target"] > 1:
                if olabels[idx1]["Target"] != olabels[idx2]["Target"]:
                    examples.append((sent_id, text, olabels[idx1]["Target"], olabels[idx2]["Polar_expression"], 0))
            if ols[idx1]["Source"] > 1:
                if olabels[idx1]["Source"] != olabels[idx2]["Source"]:
                    examples.append((sent_id, text, olabels[idx1]["Source"], olabels[idx2]["Polar_expression"], 0))
            if ols[idx2]["Target"] > 1:
                if olabels[idx1]["Target"] != olabels[idx2]["Target"]:
                    examples.append((sent_id, text, olabels[idx2]["Target"], olabels[idx1]["Polar_expression"], 0))
            if ols[idx2]["Source"] > 1:
                if olabels[idx1]["Source"] != olabels[idx2]["Source"]:
                    examples.append((sent_id, text, olabels[idx2]["Source"], olabels[idx1]["Polar_expression"], 0))
    else:
        labels = create_bio_labels(text, opinions[0])
        # convert to boolean and
        ls = {}
        for c in ["Source", "Target", "Polar_expression"]:
            labels[c] = [0 if l == "O" else 1 for l in labels[c]]
            ls[c] = len(set(labels[c]))
        # check if more than one set of labels and if so, add to examples
        if ls["Target"] > 1:
            examples.append((sent_id, text, labels["Target"], labels["Polar_expression"], 1))
        if ls["Source"] > 1:
            examples.append((sent_id, text, labels["Source"], labels["Polar_expression"], 1))
    return examples

def create_relation_jsons(filename, outfile):
    data = []
    with open(filename) as infile:
        for sent in json.load(infile):
            data.extend(convert_to_train_example(sent["sent_id"], sent["text"], sent["opinions"]))
    json_data = []
    for sent_id, text, e1, e2, label in data:
        json_data.append({"sent_id": sent_id, "text": text, "e1": e1, "e2": e2, "label": label})
    with open(outfile, "w") as o:
        for d in json_data:
            json.dump(d, o)
            o.write("\n")


def break_up_predictions(pred):
    # when there is more than one span in a prediction,
    # convert to several instances with a single span each
    #
    # No spans in prediction
    if pred.sum() == 0:
        return [pred.tolist()]
    # if everything is a span
    if pred.all() > 0:
        return [[1] * len(pred)]
    # If there are spans in prediction
    idxs = []
    bidx = None
    for i, p in enumerate(pred):
        if p > 0 and bidx is None:
            bidx = i
        if p == 0 and bidx is not None:
            idxs.append((bidx, i))
            bidx = None
        if i == len(pred) - 1 and bidx is not None:
            idxs.append((bidx, i+1))
    # convert each bidx, eidx to a np array of the size of pred
    # with 1 for each span
    preds = []
    for bidx, eidx in idxs:
        l = [0] * len(pred)
        for i in range(bidx, eidx):
            l[i] = 1
        preds.append(l)
    return preds

def break_up_expressions(pred, label2idx):
    # when there is more than one span in a prediction,
    # convert to several instances with a single span each
    #
    # No spans in prediction
    if pred.sum() == 0:
        return [pred.tolist()], []
    # if everything is a span
    if pred.all() > 0:
        full_label = label2idx.idx2label["expressions"][int(pred[0])]
        polarity = full_label.split("-")[-1]
        return [[1] * len(pred)], [polarity]
    # If there are spans in prediction
    idxs = []
    bidx = None
    polarity = None
    for i, p in enumerate(pred):
        if p > 0 and bidx is None:
            bidx = i
            full_label = label2idx.idx2label["expressions"][int(p)]
            polarity = full_label.split("-")[-1]
        if p == 0 and bidx is not None:
            idxs.append((bidx, i, polarity))
            bidx = None
            polarity = None
        if i == len(pred) - 1 and bidx is not None:
            idxs.append((bidx, i+1, polarity))
    # convert each bidx, eidx to a np array of the size of pred
    # with 1 for each span
    preds = []
    polarities = []
    for bidx, eidx, polarity in idxs:
        l = [0] * len(pred)
        for i in range(bidx, eidx):
            l[i] = 1
        preds.append(l)
        polarities.append(polarity)
    return preds, polarities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=["mpqa", "darmstadt_unis", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es"])

    args = parser.parse_args()

    for dataset in args.datasets:
        for data_split in os.listdir(os.path.join("../../data", dataset)):
            if data_split in ["train.json", "dev.json", "test.json"]:
                infile = os.path.join("../../data", dataset, data_split)
                os.makedirs(os.path.join("data", "relations", dataset),
                            exist_ok=True)
                outfile = os.path.join("data", "relations", dataset, data_split)
                create_relation_jsons(infile, outfile)
