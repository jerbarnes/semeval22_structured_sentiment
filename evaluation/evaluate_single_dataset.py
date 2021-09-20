import json
from evaluate import convert_opinion_to_tuple, tuple_f1
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")

    args = parser.parse_args()

    with open(args.gold_file) as o:
        gold = json.load(o)

    with open(args.pred_file) as o:
        preds = json.load(o)

    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])

    preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

    g = sorted(gold.keys())
    p = sorted(preds.keys())

    for i in g:
        if i not in p:
            i

    assert g == p, "missing some sentences"

    f1 = tuple_f1(gold, preds)
    print("Sentiment Tuple F1: {0:.3f}".format(f1))

if __name__ == "__main__":
    main()
