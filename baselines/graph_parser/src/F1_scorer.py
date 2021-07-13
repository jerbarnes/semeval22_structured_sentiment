import argparse
import col_data as cd
from tabulate import tabulate
import os


def get_flat(sent):
    labels = []
    for token in sent.tokens:
        scopes = token.scope
        if len(scopes) > 0:
            label = scopes[-1][-1]
        else:
            label = "O"
        labels.append(label)
    return labels

def span_f1(gold, pred, mapping, test_label="holder"):
    tp, fp, fn = 0, 0, 0
    for gold_sent, pred_sent in zip(gold, pred):
        gold_labels = get_flat(gold_sent)
        pred_labels = get_flat(pred_sent)
        for gold_label, pred_label in zip(gold_labels, pred_labels):
            gold_label = mapping[gold_label]
            pred_label = mapping[pred_label]
            # TP
            if gold_label == pred_label == test_label:
                tp += 1
            #FP
            if gold_label != test_label and pred_label == test_label:
                fp += 1
            #FN
            if gold_label == test_label and pred_label != test_label:
                fn += 1
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return prec, rec, f1

def convert_to_targeted(labeled_edges):
    targets = []
    if len(labeled_edges) == 0:
        return set(targets)
    else:
        for token_idx, edge, label in labeled_edges:
            if label == "targ":
                target = [token_idx]
                for tidx, e, l in labeled_edges:
                    if tidx == edge and "exp" in l:
                        polarity = l.split("-")[-1]
                    if e == token_idx and "targ" in l:
                        target.append(tidx)
                try:
                    targets.append((tuple(set(target)), polarity))
                # It's possible for a target not to be connected to any
                # polar expression, in which case, it will have no polarity
                except UnboundLocalError:
                    pass
        return set(targets)

def targeted_f1(gold_edges, pred_edges):
    tp, fp, fn = 0, 0, 0
    #
    for key in gold_edges.keys():
        try:
            gold_targets = convert_to_targeted(gold_edges[key])
            pred_targets = convert_to_targeted(pred_edges[key])
            tp += len(pred_targets.intersection(gold_targets))
            fp += len(pred_targets.difference(gold_targets))
            fn += len(gold_targets.difference(pred_targets))
        except:
            print(key)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return 2 * (prec * rec) / (prec + rec + 1e-6)

def get_sent_tuples(labeled_edges, keep_polarity=True):
    sent_tuples = []
    polarities = []
    expressions = []
    targets = []
    holders = []
    for token_idx, edge, label in labeled_edges:
        if edge == "0":
            polarity = label.split("-")[-1]
            polarities.append(polarity)
            exp = [token_idx]
            for t_idx, e, l in labeled_edges:
                if e == token_idx and polarity in l:
                    exp.append(t_idx)
            expressions.append(exp)
    for token_idx, edge, label in labeled_edges:
        if label == "targ":
            exp_idx = edge
            target = [token_idx]
            for t_idx, e, l in labeled_edges:
                if e in target:
                    target.append(t_idx)
            targets.append((exp_idx, target))
    for token_idx, edge, label in labeled_edges:
        if label == "holder":
            exp_idx = edge
            holder = [token_idx]
            for t_idx, e, l in labeled_edges:
                if e in holder:
                    holder.append(t_idx)
            holders.append((exp_idx, holder))
    for exp, pol in zip(expressions, polarities):
        current_targets = [t for idx, t in targets if idx == exp[0]]
        current_holders = [t for idx, t in holders if idx == exp[0]]
        if current_targets == []:
            current_targets = [[]]
        if current_holders == []:
            current_holders = [[]]
        for target in current_targets:
            for holder in current_holders:
                sent_tuples.append((frozenset(holder), frozenset(target), frozenset(exp), pol))
    return list(set(sent_tuples))

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
        if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(exp1.intersection(exp2)) > 0:
            if keep_polarity:
                if pol1 == pol2:
                    #print(holder1, target1, exp1, pol1)
                    #print(holder2, target2, exp2, pol2)
                    return True
            else:
                #print(holder1, target1, exp1, pol1)
                #print(holder2, target2, exp2, pol2)
                return True
    return False

def weighted_tuples_precision(sent_tuple1, list_of_sent_tuples):
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
        if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(exp1.intersection(exp2)) > 0:
                holder_overlap = len(holder1.intersection(holder2)) / len(holder1)
                target_overlap = len(target1.intersection(target2)) / len(target1)
                exp_overlap = len(exp1.intersection(exp2)) / len(exp1)
                return (holder_overlap + target_overlap + exp_overlap) / 3
    return 0

def weighted_score(sent_tuple1, list_of_sent_tuples):
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
        if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(exp1.intersection(exp2)) > 0:
                holder_overlap = len(holder1.intersection(holder2)) / len(holder2)
                target_overlap = len(target1.intersection(target2)) / len(target2)
                exp_overlap = len(exp1.intersection(exp2)) / len(exp2)
                return (holder_overlap + target_overlap + exp_overlap) / 3
    return 0

def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
    """
    True positives / (true positives + false positives)
    """
    tp = []
    fp = []
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        ptuples = get_sent_tuples(p)
        gtuples = get_sent_tuples(g)
        for stuple in ptuples:
                if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                    if weighted:
                        tp.append(weighted_score(stuple, gtuples))
                    else:
                        tp.append(1)
                else:
                    fp.append(1)
    return tp, fp, sum(tp) / (sum(tp) + sum(fp))

def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
    """
    True positives / (true positives + false negatives)
    """
    tp = []
    fn = []
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        ptuples = get_sent_tuples(p)
        gtuples = get_sent_tuples(g)
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    tp.append(weighted_score(stuple, ptuples))
                else:
                    tp.append(1)
            else:
                fn.append(1)
    return tp, fn, sum(tp) / (sum(tp) + sum(fn))

def tuple_F1(gold, pred, keep_polarity=True, weighted=True):
    tp, fp, prec = tuple_precision(gold, pred, keep_polarity, weighted)
    tp, fn, rec = tuple_recall(gold, pred, keep_polarity, weighted)
    return 2 * (prec * rec) / (prec + rec)

def read_labeled(file):
    """
    Read in dependency edges and labels as tuples
    (token_idx, dep_idx, label)
    """
    labeled_edges = {}
    sent_id = None
    sent_edges = []
    for line in open(file):
        if line.startswith("# sent_id"):
            sent_id = line.strip().split(" = ")[-1]
        if line.strip() == "" and sent_id is not None:
            labeled_edges[sent_id] = sent_edges
            sent_edges = []
            sent_id = None
        if line[0].isdigit():
            split = line.strip().split("\t")
            idx = split[0]
            edge_label = split[-1]
            #print(edge_label)
            if edge_label is not "_":
                if "|" in edge_label:
                    for el in edge_label.split("|"):
                        edge, label = el.split(":", 1)
                        sent_edges.append((idx, edge, label))
                else:
                    edge, label = edge_label.split(":", 1)
                    sent_edges.append((idx, edge, label))
    return labeled_edges


def read_unlabeled(file):
    """
    Read in dependency edges as tuples
    (token_idx, dep_idx)
    """
    unlabeled_edges = {}
    sent_id = None
    sent_edges = []
    for line in open(file):
        if line.startswith("# sent_id"):
            sent_id = line.strip().split(" = ")[-1]
        if line.strip() == "" and sent_id is not None:
            unlabeled_edges[sent_id] = sent_edges
            sent_edges = []
            sent_id = None
        if line[0].isdigit():
            split = line.strip().split("\t")
            idx = split[0]
            edge_label = split[-1]
            #print(edge_label)
            if edge_label is not "_":
                if "|" in edge_label:
                    for el in edge_label.split("|"):
                        edge, label = el.split(":", 1)
                        sent_edges.append((idx, edge))
                else:
                    edge, label = edge_label.split(":", 1)
                    sent_edges.append((idx, edge))
    return unlabeled_edges

def precision(gold, pred):
    """
    True positives / (true positives + false positives)
    """
    tp = 0
    fp = 0
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        for edge_label in p:
            if edge_label in g:
                tp += 1
            else:
                fp += 1
    return tp / (tp + fp)

def recall(gold, pred):
    """
    True positives / (true positives + false negatives)
    """
    tp = 0
    fn = 0
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        for edge_label in g:
            if edge_label in p:
                tp += 1
            else:
                fn += 1
    return tp / (tp + fn)

def F1(gold, pred):
    prec = precision(gold, pred)
    rec = recall(gold, pred)
    return 2 * (prec * rec) / (prec + rec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("golddir")
    parser.add_argument("preddir")
    parser.add_argument("--experiments", "-e", nargs="+", default=["point_to_root", "head_first", "head_first-inside_label", "head_final", "head_final-inside_label", "head_final-inside_label-dep_edges", "head_final-inside_label-dep_edges-dep_labels"])

    args = parser.parse_args()

    mapping = {'exp-Negative': "exp", 'exp-negative': "exp", 'IN:exp-neutral': "exp", 'exp-neutral': "exp", 'IN:exp-Negative': "exp", 'IN:exp-negative': "exp", 'targ': "targ", 'exp-positive': "exp", 'exp-Positive': "exp", 'IN:exp-Positive': "exp", 'IN:exp-positive': "exp", 'holder': "holder", 'IN:targ': "targ", 'IN:holder': "holder", 'IN:exp-None': "exp", 'exp-None': "exp", "exp-conflict": "exp", "IN:exp-conflict": "exp", "O": "O"}

    name_map = {"point_to_root": "Point-to-root",
                "head_first": "Head-first",
                "head_first-inside_label": "+inlabel",
                "head_final": "Head-final",
                "head_final-inside_label": "+inlabel",
                "head_final-inside_label-dep_edges": "Dep. edges",
                "head_final-inside_label-dep_edges-dep_labels": "Dep. labels"}


    headers = ["holder", "target", "exp", "Targeted F1", "UF", "LF", "USF", "LSF"]
    metrics = []


    # Find which experiments have been run
    #experiment_names = set(name_map.keys())
    #experiments_run = set(os.listdir(args.preddir))
    #to_check = experiment_names.intersection(experiments_run)

    for setup in args.experiments:
        metric = []
        metric.append("")
        metric.append(name_map[setup])

        #goldfile = os.path.join(args.golddir, setup, "test.conllu")
        #predfile = os.path.join(args.preddir, setup, "test.conllu.pred")
        goldfile = os.path.join(args.golddir, setup, "dev.conllu")
        predfile = os.path.join(args.preddir, setup, "dev.conllu.pred")

        gold = list(cd.read_col_data(goldfile))
        pred = list(cd.read_col_data(predfile))
        for label in ["holder", "targ", "exp"]:
            prec, rec, f1 = span_f1(gold, pred, mapping, test_label=label)
            metric.append(f1 * 100)
            #print("{0}: {1:.1f}".format(label, f1 * 100))

        lgold = read_labeled(goldfile)
        lpred = read_labeled(predfile)

        ugold = read_unlabeled(goldfile)
        upred = read_unlabeled(predfile)

        #print("Targeted F1")
        f1 = targeted_f1(lgold, lpred)
        metric.append(f1 * 100)
        #print("F1: {0:.1f}".format(f1 * 100))
        #print()

        #print("Unlabeled")
        f1 = F1(ugold, upred)
        metric.append(f1 * 100)

        #print("Labeled")
        f1 = F1(lgold, lpred)
        metric.append(f1 * 100)

        #print("Sentiment Tuple - Polarity ")
        f1 = tuple_F1(lgold, lpred, False)
        metric.append(f1 * 100)

        #print("Sentiment Tuple + Polarity")
        f1 = tuple_F1(lgold, lpred)
        metric.append(f1 * 100)

        metrics.append(metric)


    print(tabulate(metrics, headers=headers, tablefmt="latex", floatfmt="0.1f"))
