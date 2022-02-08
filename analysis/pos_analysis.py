import re
import sys
from collections import Counter

import spacy
from spacy.tokenizer import Tokenizer

from overlap_analysis import iobify, open_json, get_sequences, ROLES

ds_model_map = {
    "norec": "nb_core_news_sm",
}


def select(sequence, iobs):
    for label, iob in zip(sequence, iobs):
        if iob in "BI":
            yield label


if __name__ == "__main__":
    golds, preds, ds = open_json(sys.argv[1]), open_json(sys.argv[2]), sys.argv[3]

    gold_counts, pred_counts = Counter(), Counter()

    nlp = spacy.load(ds_model_map[ds])
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r"\S+").match)

    for role in ROLES:
        gold_sequences, pred_sequences = get_sequences(golds, preds, role)
        for gold, gold_sequence, pred_sequence in zip(golds.values(), gold_sequences, pred_sequences):
            tags = [tok.pos_ for tok in nlp(gold["text"])]
            gold_counts.update(select(tags, gold_sequence))
            pred_counts.update(select(tags, pred_sequence))
        print(role, gold_counts, pred_counts)
