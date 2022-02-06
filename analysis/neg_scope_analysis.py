import json
import sys
import argparse

from nltk.tokenize.simple import SpaceTokenizer
tk = SpaceTokenizer()

sys.path.append("../evaluation")
from evaluate import convert_opinion_to_tuple


def convert_offsets_to_set(offsets):
    """
    offsets: list of strings, each string beginning and end offset, colon-separated
    returns a set of the range from beginning to end offset + 1

    convert_offsets_to_set(["0:3"]) -> {0,1,2,3}
    """
    ranges = []
    for bidx_eidx in offsets:
        bidx, eidx = bidx_eidx.split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        ranges.extend(range(bidx, eidx+1))
    return set(ranges)

def get_neg_range(negation):
    """
    negation: list of negation annotations,
    each negation annotation is a dictionary with keys "Cue", "Scope",
    each of these is a list of strings, each string beginning and end offset, colon-separated
    returns a set of the ranges in negation from beginning to end offset + 1

    negation = {"text": "ikke tenk på det",
                "negations": [{"Cue": [["ikke"],["0:4"]],
                              "Scope": [["tenk på det"],["5:16"]]}]
                }
    get_neg_range(negation) -> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                10, 11, 12, 13, 14, 15, 16}
    """
    ranges = set()
    for ann in negation["negations"]:
        for element in ["Cue", "Scope"]:
            text, offsets = ann[element]
            ranges.update(convert_offsets_to_set(offsets))
    return ranges


def get_matching_exp(sent_tuple1, list_of_sent_tuples):
    """
    sent_tuple1: a predicted sentiment tuple
    (holder, target, expression, polarity)
    each of the subelements is a frozen set with the token offsets

    list_of_sent_tuples: a list of gold sentiment tuples

    returns any gold sentiment tuple that intersects with the prediction
    """
    matches = []
    holder1, target1, exp1, pol1 = sent_tuple1
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(exp1.intersection(exp2)) > 0:
            matches.append((holder2, target2, exp2, pol2))
    return matches


def in_neg_scope(opinion_element, neg_range):
    """
    Determine if an element of a sentiment graph (source, target, expression)
    is within the scope of some negation cue.
    """
    text, offsets = opinion_element
    off_range = convert_offsets_to_set(offsets)
    if len(off_range.intersection(neg_range)) / len(off_range) > 0.8:
        return True
    else:
        return False


def perform_analysis(sent_keys, gold_sents, pred_sents, negation_sents, element="Polar_expression"):
    """
    Count the number of predictions in the scope of negation and those out of scope.
    """


    # set up dict to keep track
    analysis_dict = {"in_neg_scope": set(),
                     "in_neg_scope_with_wrong_polarity": set(),
                     "in_neg_scope_not_predicted": set(),
                     "in_neg_scope_correct": set(),
                     "not_in_neg_scope": set(),
                     "not_in_neg_scope_with_wrong_polarity": set(),
                     "not_in_neg_scope_not_predicted": set(),
                     "not_in_neg_scope_correct": set()}

    for key in sent_keys:
        gold = gold_sents[key]
        pred = pred_sents[key]
        neg = negation_sents[key]

        neg_range = get_neg_range(neg)
        gold_tuples = convert_opinion_to_tuple(gold)
        pred_tuples = convert_opinion_to_tuple(pred)
        for opinion, tup in zip(gold["opinions"], gold_tuples):
            p_source, p_target, p_exp, p_pol, g_source, g_target, g_exp, g_pol = [None] * 8
            # IF IN SCOPE OF NEGATION
            if in_neg_scope(opinion[element], neg_range):
                analysis_dict["in_neg_scope"].add((key, tup))
                matches = get_matching_exp(tup, pred_tuples)
                g_source, g_target, g_exp, g_pol = tup
                # IF PREDICTED
                if len(matches) > 0:
                    for match in matches:
                        p_source, p_target, p_exp, p_pol = match
                        # IF POLARITY IS INCORRECT
                        if g_pol != p_pol:
                            analysis_dict["in_neg_scope_with_wrong_polarity"].add((key, g_exp, g_pol, p_exp, p_pol))
                        else:
                            analysis_dict["in_neg_scope_correct"].add((key, g_exp, g_pol, p_exp, p_pol))
                # IF NOT PREDICTED
                else:
                    analysis_dict["in_neg_scope_not_predicted"].add((key, g_exp, g_pol, p_exp, p_pol))
            # IF NOT IN SCOPE OF NEGATION
            else:
                analysis_dict["not_in_neg_scope"].add((key, tup))
                matches = get_matching_exp(tup, pred_tuples)
                g_source, g_target, g_exp, g_pol = tup
                # IF PREDICTED
                if len(matches) > 0:
                    for match in matches:
                        p_source, p_target, p_exp, p_pol = match
                        # IF POLARITY IS INCORRECT
                        if g_pol != p_pol:
                            analysis_dict["not_in_neg_scope_with_wrong_polarity"].add((key, g_exp, g_pol, p_exp, p_pol))
                        else:
                            analysis_dict["not_in_neg_scope_correct"].add((key, g_exp, g_pol, p_exp, p_pol))
                # IF NOT PREDICTED
                else:
                    analysis_dict["not_in_neg_scope_not_predicted"].add((key, g_exp, g_pol, p_exp, p_pol))
    return analysis_dict


def print_analysis(analysis_dict):
    """
    Print the analysis
    """

    print()
    print()
    print("Polar expression count:")
    print("#" * 40)
    print("In neg scope: {0}".format(len(analysis_dict["in_neg_scope"])))
    print("Not in neg scope: {0}".format(len(analysis_dict["not_in_neg_scope"])))

    neg_acc = len(analysis_dict["in_neg_scope_with_wrong_polarity"]) / len(analysis_dict["in_neg_scope"]) * 100
    non_acc = len(analysis_dict["not_in_neg_scope_with_wrong_polarity"]) / len(analysis_dict["not_in_neg_scope"]) * 100

    print()
    print()
    print("Polarity Error rate:")
    print("#" * 40)
    print("In neg scope: {0:.1f}".format(neg_acc))
    print("Not in neg scope: {0:.1f}".format(non_acc))


    neg_recall = (len(analysis_dict["in_neg_scope_with_wrong_polarity"]) + len(analysis_dict["in_neg_scope_correct"])) / len(analysis_dict["in_neg_scope"])
    non_recall = (len(analysis_dict["not_in_neg_scope_with_wrong_polarity"]) + len(analysis_dict["not_in_neg_scope_correct"])) / len(analysis_dict["not_in_neg_scope"])
    print()
    print("Recall:")
    print("#" * 40)
    print("In neg scope: {0:.3f}".format(neg_recall))
    print("Not in neg scope: {0:.3f}".format(non_recall))



def get_args():
    """
    Helper function to get the gold json, predictions json and negation jsons
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("predictions")
    parser.add_argument("negation_scope")
    args = parser.parse_args()
    return args


def open_json(json_file):
    """
    Helper function to open the json files
    """
    with open(json_file) as o:
        file = json.load(o)
    sent_dict = dict([(sent["sent_id"], sent) for sent in file])
    sent_keys = set(sent_dict.keys())
    return sent_keys, sent_dict


def main():
    args = get_args()

    gold_keys, gold = open_json(args.gold)
    pred_keys, pred = open_json(args.predictions)
    neg_keys, neg = open_json(args.negation_scope)

    # there are 4 sents in Norec_Fine that are not found in Norec_Neg
    # and vice versa, so we take the intersection and work with these
    final_keys = gold_keys.intersection(neg_keys)

    # keys of sents with negation
    neg_keys = [x["sent_id"] for x in neg.values() if len(x["negations"]) > 0]

    analysis_dict = perform_analysis(final_keys, gold, pred, neg, "Polar_expression")
    print_analysis(analysis_dict)


if __name__ == "__main__":
    main()
