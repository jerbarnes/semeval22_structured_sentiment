# click is imported in metrics.py, but we don't need it, so just make sure it
# doesn't crash
import sys
from unittest.mock import MagicMock
sys.modules.setdefault("click", MagicMock())
import count_errors
from domain_analysis import open_json
# we don't need the keys
open_json = lambda file, open_json=open_json: open_json(file)[1]

ROLES = ["Source", "Target", "Polar_expression"]

def token_ids_from_offsets(tokens, offsets, text=None):
    token_ids = set()
    for start, stop in offsets:
        if stop < 0:
            # convert negative indices to length of text plus index
            stop += sum(len(token) for token in tokens) + len(tokens) - 1
        cindex = tindex = 0
        while len(tokens) > tindex:
            # do we start before this token?
            if start == cindex:
                tstart = tindex
            # "consume" token
            cindex += len(tokens[tindex])
            tindex += 1
            # do we stop after this token?
            if stop == cindex:
                token_ids.update(range(tstart, tindex))
                break
            # consume whitespace
            cindex += 1
        else:
            # we never broke out of the loop: no matching end found at token
            # boundary
            # if text:
            #     print("Oh no.", repr(text), repr(text[start:stop]))
            raise RuntimeError(f"No match for {start}:{stop} in {tokens}")
    return token_ids

def iobify(instance, role):
    tokens = instance["text"].split(" ")
    marked_tokens = set()
    for opinion in instance["opinions"]:
        _, offsets = opinion[role]
        offsets = [[int(part) for part in offset.split(":")] for offset in offsets]
        try:
            marked_tokens.update(token_ids_from_offsets(tokens, offsets, text=instance["text"]))
        except RuntimeError:
            print("Skipping opinion with invalid offsets")
    result = []
    state = "O"
    for i, token in enumerate(tokens):
        if i in marked_tokens:
            if state == "O":
                result.append("B")
            else:
                result.append("I")
        else:
            result.append("O")
        state = result[-1]
    return result

def get_sequences(golds, preds, role):
    gold_sequences, pred_sequences = [], []
    for sent_id, gold_sent in golds.items():
        # if you don't annotate an instance, we punish by adding an empty
        # annotation. Otherwise we would let submissions "cheat" by ommitting
        # difficult instances:
        pred_sent = preds.get(sent_id, {"text": gold_sent["text"], "opinions": []})
        gold_sequences.append(iobify(gold_sent, role))
        pred_sequences.append(iobify(pred_sent, role))
    return gold_sequences, pred_sequences

if __name__ == "__main__":
    golds, preds = map(open_json, sys.argv[1:])

    # transform into form expected by metrics (token-based IOB)
    for role in ROLES:
        gold_sequences, pred_sequences = get_sequences(golds, preds, role)
        print(role)
        print(count_errors.error_classes(gold_sequences, pred_sequences))
