import json
import sys
import argparse

sys.path.append("../evaluation")
from evaluate import tuple_f1, convert_opinion_to_tuple


def get_args():
    """
    Helper function to get the gold json, predictions json and negation jsons
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("predictions")
    parser.add_argument("metadata")
    args = parser.parse_args()
    return args


def open_json(json_file):
    """
    Helper function to open the json files
    """
    with open(json_file) as o:
        file = json.load(o)
    sent_dict = {sent["sent_id"]: sent for sent in file}
    sent_keys = set(sent_dict.keys())
    return sent_keys, sent_dict


def main():
    args = get_args()
    with open(args.metadata) as o:
        metadata = json.load(o)

    test_domains = {}

    gold_keys, gold = open_json(args.gold)
    pred_keys, pred = open_json(args.predictions)

    # get the domains found in the test data
    for sent_id in gold_keys:
        domain = metadata[sent_id[:6]]["category"]
        if domain not in test_domains:
            test_domains[domain] = [sent_id]
        else:
            test_domains[domain].append(sent_id)

    # print the domains in descending order
    for key, value in sorted(test_domains.items(), key=lambda kv: len(kv[1])):
        print("{}:     \t{}".format(key, len(value)))
    print()
    print()

    # get the sentiment graph F1 for each domain
    for domain, sent_ids in sorted(test_domains.items(),
                                   key=lambda kv: len(kv[1])):
        domain_gold = dict([(sent_id, convert_opinion_to_tuple(gold[sent_id]))                for sent_id in sent_ids])
        domain_pred = dict([(sent_id, convert_opinion_to_tuple(pred[sent_id]))                for sent_id in sent_ids])
        f1 = tuple_f1(domain_gold, domain_pred)
        print("{0}: {1:.3f}".format(domain, f1))


if __name__ == "__main__":
    main()
