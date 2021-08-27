import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader


import numpy as np
import argparse
import os
import pickle
import json
from copy import copy

from utils import SetVocab, ExtractionInferenceDataset, get_best_run, get_offsets, convert_prediction_to_token_ids
from convert_to_rels import break_up_predictions, break_up_expressions

from extraction_module import Bilstm
from relation_prediction_module import Relation_Model, find_best_rel_model


def extract_spans(model_dir, datafile, args):

    predictions = {"sources": [],
                   "targets": [],
                   "expressions": []
                   }

    for annotation in ["sources", "targets", "expressions"]:

        basedir = os.path.join(model_dir,
                               annotation)

        print("opening {} model...".format(annotation))
        with open(os.path.join(basedir,
                               "params.pkl"), "rb") as infile:
            params = pickle.load(infile)

        (w2idx,
         matrix_shape,
         num_labels,
         label2idx,
         hidden_dim,
         num_layers,
         annotation) = params

        extraction_vocab = SetVocab(w2idx)

        extraction_dataset = ExtractionInferenceDataset(extraction_vocab,
                                                        label2idx,
                                                        False)

        inference_data = extraction_dataset.get_split(datafile,
                                                      lower_case=False,
                                                      annotation=annotation
                                                      )

        tag2idx = label2idx.label2idx[annotation]

        dev_loader = DataLoader(inference_data,
                                batch_size=args.BATCH_SIZE,
                                collate_fn=inference_data.collate_fn,
                                shuffle=False)

        new_matrix = np.zeros(matrix_shape)

        model = Bilstm(extraction_vocab,
                       new_matrix,
                       num_labels,
                       tag2idx,
                       matrix_shape[1],
                       hidden_dim,
                       num_layers,
                       train_embeddings=False)

        f1, params, best_weights = get_best_run(basedir)
        model.load_state_dict(torch.load(best_weights))
        model.eval()

        sent_ids, pred = model.predict(dev_loader)
        predictions[annotation] = pred
    return sent_ids, predictions, label2idx


def predict_relations(model_dir, datafile, predictions, sent_ids, label2idx, args):
    opinions = []

    with open(os.path.join(model_dir,
                           "params.pkl"), "rb") as infile:
            params = pickle.load(infile)

    (w2idx,
     matrix_shape,
     HIDDEN_DIM,
     NUM_LAYERS,
     POOLING) = params
    rel_vocab = SetVocab(w2idx)
    new_matrix = np.zeros(matrix_shape)

    # create relation prediction model
    rel_model = Relation_Model(rel_vocab,
                               embedding_dim=matrix_shape[1],
                               hidden_dim=HIDDEN_DIM,
                               embedding_matrix=new_matrix,
                               pooling=POOLING)

    print("loading best relation prediction model...")
    f1, best_model = find_best_rel_model(model_dir)
    rel_model.load_state_dict(torch.load(best_model))

    with open(datafile) as infile:
        original_data = json.load(infile)
    data_dict = dict([(item["sent_id"], item) for item in original_data])

    for i, sent_id in enumerate(sent_ids):
        text = data_dict[sent_id]["text"]
        json_entry = {"sent_id": sent_id, "text": text, "opinions": []}
        token_dict = dict([(i, t) for i, t in enumerate(text.split())])
        offset_dict = get_offsets(text)
        seq = pack_sequence(torch.LongTensor(rel_vocab.ws2ids(text.split())).unsqueeze(0))
        #seq = pack_sequence(inference_data[i][1].unsqueeze(0))
        sources = break_up_predictions(predictions["sources"][i].numpy())
        targets = break_up_predictions(predictions["targets"][i].numpy())
        expressions, polarities = break_up_expressions(predictions["expressions"][i].numpy(), label2idx)

        for expression, polarity in zip(expressions, polarities):
            added_to_opinions = False
            opinion = {"Source": [[], []],
                       "Target": [[], []],
                       "Polar_expression": [[], []],
                       "Polarity": None,
                       "Intensity": "Standard"
                       }
            # If there is no expression just move on
            if np.sum(expression) == 0:
                continue
            else:
                # get expression tokens and offsets
                tokens, offset = get_tokens_offsets(expression,
                                                    token_dict,
                                                    offset_dict)
                opinion["Polar_expression"] = [[tokens], [offset]]
                opinion["Polarity"] = polarity

                packed_expression = pack_sequence(torch.LongTensor(expression).unsqueeze(0))
            for source in sources:
                if np.sum(source) == 0:
                    pass
                else:
                    batch_sizes = torch.LongTensor([1])
                    packed_source = pack_sequence(torch.LongTensor(source).unsqueeze(0))
                    pred = rel_model.forward(seq,
                                             packed_source,
                                             packed_expression,
                                             batch_sizes)
                    # If we predict a relationship
                    if pred[0] > 0.4:
                        # get expression tokens and offsets
                        tokens, offset = get_tokens_offsets(source,
                                                            token_dict,
                                                            offset_dict)
                        opinion["source"] = [[tokens], [offset]]
                for target in targets:
                    if np.sum(target) == 0:
                        pass
                    else:
                        batch_sizes = torch.LongTensor([1])
                        packed_target = pack_sequence(torch.LongTensor(target).unsqueeze(0))
                        pred = rel_model.forward(seq,
                                                 packed_target,
                                                 packed_expression,
                                                 batch_sizes)
                        # If we predict a relationship
                        if pred[0] > 0.4:
                            # get expression tokens and offsets
                            tokens, offset = get_tokens_offsets(target,
                                                                token_dict,
                                                                offset_dict)
                            # expression can have several targets
                            o = copy(opinion)
                            o["Target"] = [[tokens], [offset]]
                            json_entry["opinions"].append(o)
                            added_to_opinions = True
                if not added_to_opinions:
                    json_entry["opinions"].append(opinion)

        opinions.append(json_entry)

    return opinions

def get_tokens_offsets(pred, token_dict, offset_dict):
    idxs = convert_prediction_to_token_ids(pred)
    tokens = " ".join([token_dict[t] for t in idxs])
    offsets = [offset_dict[t] for t in idxs]
    bidx = offsets[0][0]
    eidx = offsets[-1][-1]
    offset = "{0}:{1}".format(bidx, eidx)
    return tokens, offset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATADIR", "-data", default="opener_en")
    parser.add_argument("--FILE", "-file", default="dev.json")
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)

    args = parser.parse_args()
    print(args)

    ##################################################
    # Extract spans
    ##################################################

    extraction_model_dir = os.path.join("saved_models",
                                        "extraction_models",
                                        args.DATADIR)

    datafile = os.path.join("../../data",
                            args.DATADIR,
                            args.FILE)

    sent_ids, span_predictions, label2idx = extract_spans(extraction_model_dir,
                                                          datafile,
                                                          args)

    ##################################################
    # Predict relations
    ##################################################

    # Load relation prediction model
    relation_model_dir = os.path.join("saved_models",
                                      "relation_prediction",
                                      args.DATADIR
                                      )

    full_predictions = predict_relations(relation_model_dir,
                                         datafile,
                                         span_predictions,
                                         sent_ids,
                                         label2idx,
                                         args)

    with open(os.path.join(relation_model_dir, "prediction.json"), "w") as o:
        json.dump(full_predictions, o)


if __name__ == "__main__":
    main()
