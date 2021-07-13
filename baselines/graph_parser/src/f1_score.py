import numpy as np

def tps(gold, pred, label):
    tp, fp, fn = 0, 0, 0
    for g, p in zip(gold, pred):
        if g == label and g == p:
            tp += 1
        elif p == label and g != p:
            fp += 1
        elif g == label and g != p:
            fn += 1
    return tp, fp, fn

def f1_score(gold, pred):
    tp, fp, fn = tps(gold, pred, 1)

if __name__ == "__main__":

    devo, predo, devt, predt = [], []

    for line in open("data/norec/dev/opinion.txt"):
        devo.extend(np.array(line.strip().split(), dtype=int))

    for line in open("predictions/norec/dev/opinions.txt"):
        predo.extend(np.array(line.strip().split(), dtype=int))

    for line in open("data/norec/dev/target.txt"):
        devt.extend(np.array(line.strip().split(), dtype=int))

    for line in open("predictions/norec/dev/target.txt"):
        predt.extend(np.array(line.strip().split(), dtype=int))

