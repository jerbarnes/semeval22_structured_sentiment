import json
import os
import argparse
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
            polarity = opinion["Polarity"]
            target_tokens = t.split()
            label = "-targ-{0}".format(polarity)
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = text[0].split()
        label = "-targ-{0}".format(polarity)
        #
        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
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
            label = "-exp-{0}".format(polarity)
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = text[0].split()
        label = "-exp-{0}".format(polarity)
        #
        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
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
            label = "-holder"
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = text[0].split()
        label = "-holder"
        #
        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
        return [(bidx, tags)]


def replace_with_labels(labels, offsets, bidx, tags):
    # There are some annotations that missed token level (left out a leading character) that we need to fix
    try:
        token_idx = offsets.index(bidx)
        for i, tag in enumerate(tags):
            labels[i + token_idx] = tag
        return labels
    except:
        return labels

def restart_orphans(labels):
    """Wen opinion expression tags are written on top of previous expression tags,
    I-tags can be orphaned, so they do not correspond with the previous tag. We reset these to a B

        labels : list(Str) tag sequence for a sentence.
    """
    prev = "O"
    for tag_idx,tag in enumerate(labels):
        if tag[0] == "I":
            if prev == "O" or (len(prev)>1 and tag[1:] != prev[1:]):
                labels[tag_idx] = "B"+tag[1:] #Replace I with B since contents is different from prev
                #print("correcting", prev, tag)
        prev = labels[tag_idx]
    return labels


def create_bio_labels(text, opinions):
    offsets = [l[0] for l in tk.span_tokenize(text)]
    #
    columns = ["Source", "Target", "Polar_expression"]
    labels = {c: ["O"] * len(offsets) for c in columns}
    #
    anns = {c: [] for c in columns}


    # TODO: deal with targets which can have multiple polarities, due to
    # contrasting polar expressions. At present the last polarity wins.
    for o in opinions:
        try:
            anns["Source"].extend(get_bio_holder(o))
        except:
            pass
        try:
            anns["Target"].extend(get_bio_target(o))
        except:
            pass
        try:
            anns["Polar_expression"].extend(get_bio_expression(o))
        except:
            pass
    #
    for c in columns:
        for bidx, tags in anns[c]:
            labels[c] = replace_with_labels(labels[c], offsets, bidx, tags)
        labels[c] = restart_orphans(labels[c])
    return labels


def to_bio(dataset):
    bio_dataset = []
    for i, sent in enumerate(dataset):
        idx = sent["sent_id"]
        text = sent["text"]
        opinions = sent["opinions"]
        tokens = text.split()
        labels = create_bio_labels(text, opinions)
        bio_sent = {"sent_id": idx,
                    "text": tokens,
                    "sources": labels["Source"],
                    "targets": labels["Target"],
                    "expressions": labels["Polar_expression"]
                    }
        bio_dataset.append(bio_sent)
    return bio_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=["mpqa", "darmstadt_unis", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es"])

    args = parser.parse_args()

    for dataset in args.datasets:
        for data_split in os.listdir(os.path.join("../../data", dataset)):
            if data_split in ["train.json", "dev.json", "test.json"]:
                with open(os.path.join("../../data", dataset, data_split)) as o:
                    split = json.load(o)
                bio_split = to_bio(split)
                os.makedirs(os.path.join("data", "extraction", dataset), exist_ok=True)
                with open(os.path.join("data", "extraction", dataset, data_split), "w") as outfile:
                    for example in bio_split:
                        json.dump(example, outfile)
                        outfile.write("\n")


