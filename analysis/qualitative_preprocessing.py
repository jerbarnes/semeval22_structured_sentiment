import sys
sys.path.append("../evaluation")
from evaluate import convert_char_offsets_to_token_idxs
from domain_analysis import open_json

from nltk.tokenize.simple import SpaceTokenizer
import os

tk = SpaceTokenizer()

def corpus_predictions(corpus,predictions_path,setup):
    """
    Creates dictionary with all predictions for a corpus
    """
    mydict = {}

    # iterate through teams
    for s in os.listdir(predictions_path):
        current_path = os.path.join(predictions_path, s+"/"+setup)
        if "ohhhmygosh" in s and setup=="crosslingual":
                continue

        if "submission" in s and not s.endswith(".zip"):

            pred_file = os.path.join(current_path, corpus+"/predictions.json")
            pred_keys, pred = open_json(pred_file)
            check_predicted = [pred[key]["opinions"] for key in pred_keys if len(pred[key]["opinions"])!=0]

            if len(check_predicted) != 0:

                mydict[s]= pred

    return mydict


def check_alignment_condition(funct, holder1, target1, exp1,holder2, target2, exp2):
    if "exact" in funct:
        if (
            holder2 == holder1
            and target2 == target1
            and len(exp2.intersection(exp1)) > 0
        ):
            return True
    else:
        if (
            len(holder2.intersection(holder1)) > 0
            and len(target2.intersection(target1)) > 0
            and len(exp2.intersection(exp1)) > 0
        ):
            return True


def align_gold_pred(funct,a_gold, all_preds):
    """
    (adapted from function weighted_score in evaluate.py)
    """
    best_overlap = 0
    holder1, target1, exp1, pol1, intens1,txt = a_gold
    for holder2, target2, exp2, pol2, intens2,txt in all_preds:
        if check_alignment_condition (funct,holder1, target1, exp1,holder2, target2, exp2):
            
            holder_overlap = len(holder1.intersection(holder2)) / len(holder2)
            target_overlap = len(target1.intersection(target2)) / len(target2)
            exp_overlap = len(exp1.intersection(exp2)) / len(exp2)
            overlap = (holder_overlap + target_overlap + exp_overlap) / 3

            if overlap > best_overlap:
                best_overlap = overlap
                match = [holder2, target2, exp2, pol2, intens2]

    if best_overlap==0:
        match = []

    return match


def opinion_to_tuple(sentence):

    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = list(tk.span_tokenize(text))

    if len(opinions) > 0:
        txt = []
        for offs in token_offsets:
            o = [":".join(map(str,offs))]
            txt.append(convert_char_offsets_to_token_idxs(o, token_offsets))

        for opinion in opinions:
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]
            polarity = opinion["Polarity"]
            try:
                intensity = opinion["Intensity"]
            except KeyError: # no intensity predicted
                intensity = ""
            
            if str(intensity)=="None":
                intensity="N0NE"  

            holder = convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            if len(holder) == 0:
                holder = frozenset(["_"])
            if len(target) == 0:
                target = frozenset(["_"])
            opinion_tuples.append((holder, target, exp, polarity, intensity,txt))

    return opinion_tuples
