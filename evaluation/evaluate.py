#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function, division
import sys
import os
import json

from nltk.tokenize.simple import SpaceTokenizer

tk = SpaceTokenizer()


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


def convert_opinion_to_tuple(sentence):
    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = list(tk.span_tokenize(text))
    #
    if len(opinions) > 0:
        for opinion in opinions:
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]
            polarity = opinion["Polarity"]
            #
            holder = convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            opinion_tuples.append((holder, target, exp, polarity))
    return opinion_tuples


def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder1.intersection(holder2)) > 0
            and len(target1.intersection(target2)) > 0
            and len(exp1.intersection(exp2)) > 0
        ):
            if keep_polarity:
                if pol1 == pol2:
                    # print(holder1, target1, exp1, pol1)
                    # print(holder2, target2, exp2, pol2)
                    return True
            else:
                # print(holder1, target1, exp1, pol1)
                # print(holder2, target2, exp2, pol2)
                return True
    return False


def weighted_score(sent_tuple1, list_of_sent_tuples):
    best_overlap = 0
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder2.intersection(holder1)) > 0
            and len(target2.intersection(target1)) > 0
            and len(exp2.intersection(exp1)) > 0
        ):
            holder_overlap = len(holder2.intersection(holder1)) / len(holder1)
            target_overlap = len(target2.intersection(target1)) / len(target1)
            exp_overlap = len(exp2.intersection(exp1)) / len(exp1)
            overlap = (holder_overlap + target_overlap + exp_overlap) / 3
            if overlap > best_overlap:
                best_overlap = overlap
    return best_overlap


def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                if weighted:
                    #sc = weighted_score(stuple, gtuples)
                    #if sc != 1:
                        #print(sent_idx)
                        #print(sc)
                        #print()
                    weighted_tp.append(weighted_score(stuple, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                #print(sent_idx)
                fp.append(1)
    #print("weighted tp: {}".format(sum(weighted_tp)))
    #print("tp: {}".format(sum(tp)))
    #print("fp: {}".format(sum(fp)))
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)


def tuple_f1(gold, pred, keep_polarity=True, weighted=True):
    prec = tuple_precision(gold, pred, keep_polarity, weighted)
    rec = tuple_recall(gold, pred, keep_polarity, weighted)
    #print("prec: {}".format(prec))
    #print("rec: {}".format(rec))
    return 2 * (prec * rec) / (prec + rec + 0.00000000000000001)


def main():
    """
    Evaluate monolingual structured sentiment results.
    """
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Paths correspond to what Codalab expects
    submit_dir = os.path.join(input_dir, "res/")
    truth_dir = os.path.join(input_dir, "ref/data")

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, "scores.txt")
    output_file = open(output_filename, "w")

    monolingual_datasets = [
        "norec",
        "multibooked_ca",
        "multibooked_eu",
        "opener_en",
        "opener_es",
        "mpqa",
        "darmstadt_unis",
    ]
    crosslingual_datasets = [
        "opener_es",
        "multibooked_ca",
        "multibooked_eu"
        ]

    for subtask, datasets in [("monolingual", monolingual_datasets),
                              ("crosslingual", crosslingual_datasets)]:
        results = []

        print("{}".format(subtask))
        print("#" * 40)

        for dataset in datasets:
            gold_file = os.path.join(truth_dir, subtask, dataset, "test.json")
            submission_answer_file = os.path.join(submit_dir, subtask, dataset,  "predictions.json")

            # read in gold and predicted data, convert to dictionaries
            # where the sent_ids are keys
            with open(gold_file) as infile:
                gold = json.load(infile)
            gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])

            with open(submission_answer_file) as infile:
                preds = json.load(infile)
            preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

            # make sure they have the same keys
            # Todo: make the error message more useful by including the missing values

            g = set(gold.keys())
            p = set(preds.keys())

            assert g.issubset(p), "missing some sentences: {}".format(g.difference(p))
            assert p.issubset(g), "predictions contain sentences that are not in golds: {}".format(p.difference(g))

            f1 = tuple_f1(gold, preds)
            results.append(f1)
            print("SF1 on {0}: {1:.3f}".format(dataset, f1))
            if subtask == "crosslingual":
                crossdataset = "cross_" + dataset
                output_file.write("{0}: {1:.3f}\n".format(crossdataset, f1))
            else:
                output_file.write("{0}: {1:.3f}\n".format(dataset, f1))

        ave_score = sum(results) / len(results)
        print("Average score: {:.3f}".format(ave_score))
        print()

        if subtask == "crosslingual":
            output_file.write("cross_ave_score: {:.3f}\n".format(ave_score))
        else:
            output_file.write("ave_score: {:.3f}\n".format(ave_score))


if __name__ == "__main__":
    main()
