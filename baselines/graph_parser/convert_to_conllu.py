import stanza
import json
import argparse
import os

from data_utils import tokenidx2edge, tokenidx2deplabel, create_labels, create_sentiment_dict, redefine_root_with_dep_edges, combine_sentiment_dicts, create_conll_sent_dict


def stanza_tag_json_files(json_file):
    for sentence in json_file:
        tagged_sent = nlp(sentence["text"])
        conllu = ""
        for sent in tagged_sent.sentences:
            for i, token in enumerate(sent.words):
                # ID  TOKEN  LEMMA  UPOS  XPOS  MORPH  HEAD_ID  DEPREL
                conllu += "{}\t{}\t{}\t{}\t{}\t_\t{}\t{}\t_\t_\n".format(i+1, token.text, token.lemma, token.pos, "_", token.head, token.deprel)
        sentence["conllu"] = conllu


def get_sent_conllus(json_file,
                     setup="point_to_root",
                     inside_label=False,
                     use_dep_edges=False,
                     use_dep_labels=False):
    for idx, sentence in enumerate(json_file):
        try:
            sentiment_conllu = ""
            sent_id = sentence["sent_id"]
            text = sentence["text"]
            sentiment_conllu += "# sent_id = {}\n".format(sent_id)
            sentiment_conllu += "# text = {}\n".format(text)
            opinions = sentence["opinions"]
            conllu = sentence["conllu"]
            t2e = tokenidx2edge(conllu)
            t2l = tokenidx2deplabel(conllu)

            if len(opinions) > 0:
                labels = [create_labels(text, o) for o in opinions]
            else:
                labels = [create_labels(text, [])]
            #
            sent_labels = []
            for l in labels:
                try:
                    sent_labels.append(create_sentiment_dict(l,
                                       setup=setup,
                                       inside_label=inside_label
                                       ))
                except UnboundLocalError:
                    # UnboundLocalError: local variable 'exp_root_id' referenced before assignment
                    # This error happens if there is no sentiment expression in the opinion, which should not happen
                    pass

            # Double check that the sent_labels dictionary is not empty
            if len(sent_labels) == 0:
                sent_labels = [create_sentiment_dict(create_labels(text, []),
                                                     setup=setup,
                                                     inside_label=inside_label)]

            if use_dep_edges:
                if use_dep_labels:
                    sent_labels = [redefine_root_with_dep_edges(s, t2e, t2l) for s in sent_labels]
                else:
                    sent_labels = [redefine_root_with_dep_edges(s, t2e) for s in sent_labels]

            combined_labels = combine_sentiment_dicts(sent_labels)
            conll = create_conll_sent_dict(conllu)
            for i in conll.keys():
                if i not in combined_labels:
                    combined_labels[i] = "_"
                sentiment_conllu += conll[i] + "\t" + combined_labels[i] + "\n"
            sentence["sentiment_conllu"] = sentiment_conllu
        except:
            print(idx)

def print_sentconllu(jsonfile, outfile):
    with open(outfile, "w") as o:
        for sent in jsonfile:
            try:
                o.write(sent["sentiment_conllu"] + "\n")
            # does not have a sentiment conllu due to some previous error
            except KeyError:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="opener_es")
    parser.add_argument("--out_dir", default="sentiment_graphs/opener_es")
    parser.add_argument("--setup", default="head_final")
    parser.add_argument("--inside_label", action="store_true")
    parser.add_argument("--use_dep_edges", action="store_true")
    parser.add_argument("--use_dep_labels", action="store_true")

    args = parser.parse_args()

    dataset_to_lang = {"multibooked_eu": "eu",
                       "multibooked_ca": "ca",
                       "opener_en": "en",
                       "opener_es": "es",
                       "norec": "no",
                       "mpqa": "en",
                       "darmstadt_unis": "en"}

    dataset = os.path.basename(args.json_dir)
    nlp = stanza.Pipeline(dataset_to_lang[dataset],
                          processors='tokenize,pos,lemma,depparse',
                          tokenize_no_ssplit=True,
                          tokenize_pretokenized=True)

    tag_json_files = stanza_tag_json_files

    print("Dataset: {}".format(args.json_dir))
    print("Setup: {}".format(args.setup))
    if args.inside_label:
        print("Using Inside Label")
    if args.use_dep_edges:
        print("Using Dependency Edges to create sentiment graph")
    if args.use_dep_labels:
        print("Using Dependency Labels to create sentiment graph")


    out_dir = os.path.join(args.out_dir, args.setup)
    if args.inside_label:
        out_dir += "-inside_label"
    if args.use_dep_edges:
        out_dir += "-dep_edges"
    if args.use_dep_labels:
        out_dir += "-dep_labels"
    os.makedirs(out_dir, exist_ok=True)

    to_convert = [file for file in os.listdir(args.json_dir) if file in ["train.json", "dev.json", "test.json"]]

    for file in to_convert:
        with open(os.path.join(args.json_dir, file)) as infile:
            train = json.load(infile)
        tag_json_files(train)
        get_sent_conllus(train,
                         setup=args.setup,
                         inside_label=args.inside_label,
                         use_dep_edges=args.use_dep_edges,
                         use_dep_labels=args.use_dep_labels)
        print_sentconllu(train, os.path.join(out_dir,
                                             file.replace("json", "conllu")))
