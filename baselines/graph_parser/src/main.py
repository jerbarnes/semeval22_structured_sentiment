import torch
import argparse
from argparse import ArgumentParser, Namespace
from model_interactor import ModelInteractor
#import vocab as vcb
from vocab import Vocab, Vocabs, make_vocabs
import col_data as cd
import scorer as sc
import pickle
import cfg_parser
import numpy as np
import json

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(forced_args=None):
    parser = ArgumentParser(
        description="Options for the Neural Semantic Dependency Parser")
    parser.add_argument("--cont", action="store_true")
    parser.add_argument("--config", metavar="FILE")
    parser.add_argument("--dir", type=str, default="./")
    parser.add_argument("--elmo_train", type=str, default=None)
    parser.add_argument("--elmo_dev", type=str, default=None)
    parser.add_argument("--elmo_test", type=str, default=None)
    parser.add_argument("--vec_dim", type=int, default=1024)
    parser.add_argument("--use_elmo", type=str2bool, default=False)
    parser.add_argument("--recycle", type=str, default=None)
    parser.add_argument("--recycle_layers", type=str, default=None)
    parser.add_argument("--freeze", type=str, default=None)
    parser.add_argument("--tree", type=str2bool, default=False)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument(
        "--train",
        help="train file",
        metavar="FILE",
        )#default="")
    parser.add_argument(
        "--val",
        help="validation file",
        metavar="FILE",
        )#default="")
    parser.add_argument(
        "--external",
        help="Glove file",
        metavar="FILE",
        )#default="data/glove.6B.100d.txt")
    parser.add_argument("--batch_size", type=int, )#default=50)
    parser.add_argument("--gcn_layers", type=int, )#default=50)
    parser.add_argument("--epochs", type=int, )#default=70)
    parser.add_argument(
        "--hidden_lstm",
        help="The dimension of the hidden state in the LSTMs",
        type=int,
        )#default=600)
    parser.add_argument(
        "--hidden_char_lstm",
        help="The dimension of the hidden state in the char LSTMs",
        type=int,
        )#default=400)
    parser.add_argument("--layers_lstm", type=int, )#default=3)
    parser.add_argument(
        "--dim_mlp",
        help="Out dimension of the mlp transforming the LSTM outputs",
        type=int,
        )#default=600)
    parser.add_argument("--dim_embedding", type=int, )#default=100)
    parser.add_argument("--dim_char_embedding", type=int, )#default=100)
    parser.add_argument(
        "--bridge",
        type=str,
        choices=["dpa", "dpa+", "gcn", "simple"],
        )#default="xpos")
    parser.add_argument(
        "--pos_style",
        type=str,
        choices=["upos", "xpos"],
        )#default="xpos")
    parser.add_argument(
        "--target_style",
        type=str,
        choices=["sem", "syn", "scope", "scope-"],
        )#default="sem")
    parser.add_argument(
        "--other_target_style",
        type=str,
        choices=["sem", "syn", "scope", "none"],
        )#default="sem")
    parser.add_argument(
        "--help_style",
        type=str,
        )#default="sem")
    parser.add_argument(
        "--attention",
        type=str,
        choices=["biaffine", "bilinear", "affine"],
        )#default="biaffine")
    parser.add_argument(
        "--model_interpolation",
        help=
        "model-inter = interpolation * other_loss + (1-interpolation) * primary_loss",
        type=float,
        )#default=0.5)
    parser.add_argument(
        "--loss_interpolation",
        help=
        "Loss = interpolation * label_loss + (1-interpolation) * edge_loss",
        type=float,
        )#default=0.025)
    parser.add_argument(
        "--lstm_implementation",
        type=str,
        choices=["drop_connect", "native"],
        #default="drop_connect",
        help=
        "drop_connect uses stacked native layers which are then monkey patched to facilitate recurrent dropout. "
        +
        "native is completely using built implementation, ignoring recurrent dropout."
    )
    parser.add_argument(
        "--char_implementation",
        type=str,
        choices=["convolved", "single"],
        #default="convolved",
        help="convolved convolves in each time step over 3 character embeddings"
        + "single uses a single raw character embedding in each time step")
    dropout_group = parser.add_argument_group(
        "Dropout", "Decide how much dropout to apply where")
    dropout_group.add_argument(
        "--dropout_embedding",
        help="The chance of a word type being randomly dropped",
        type=float,
        )#default=0.2)
    dropout_group.add_argument(
        "--dropout_edge",
        help="Dropout to edge output (before attention)",
        type=float,
        )#default=0.25)
    dropout_group.add_argument(
        "--dropout_label",
        help="Dropout to label output (before attention)",
        type=float,
        )#default=0.33)
    dropout_group.add_argument(
        "--dropout_main_recurrent",
        help="Dropout in main LSTMs between time steps",
        type=float,
        )#default=0.25)
    dropout_group.add_argument(
        "--dropout_recurrent_char",
        help="Dropout in char LSTMs between time steps",
        type=float,
        )#default=0.33)
    dropout_group.add_argument(
        "--dropout_main_ff",
        help="Dropout in the in feed forward direction of the main LSTM",
        type=float,
        )#default=0.45)
    dropout_group.add_argument(
        "--dropout_char_ff",
        help="Dropout in the in feed forward direction of the char LSTM",
        type=float,
        )#default=0.33)
    dropout_group.add_argument(
        "--dropout_char_linear",
        help=
        "Dropout in the FNN transforming char LSTM output to the embedding dimension",
        type=float,
        )#default=0.33)
    parser.add_argument(
        "--disable_external",
        help="Disables the use of external embeddings",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--disable_char",
        help="Disables the use of character embeddings",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--disable_lemma",
        help="Disables the use of lemma embeddings",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--disable_pos",
        help="Disables the use of part of speech embeddings",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--disable_form",
        help="Disables the use of the word form embeddings",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--seed", help="Sets the random seed", type=int, )#default=1234)
    parser.add_argument(
        "--force_cpu",
        help="Uses CPU even if GPU is available.",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--quiet",
        help="Disables the between batch update notifications",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--save_every",
        help=
        "Saves a different copy after each epoch. May consume a lot of disk space!"
        +
        "Default behavior is to overwrite a single file with the new updates.",
        type=str2bool,
        )#action="store_true")
    optimizer = parser.add_argument_group(
        "Optimizer", "Set the Adam optimizer hyperparameters")
    optimizer.add_argument(
        "--beta1",
        help="Tunes the running average of the gradient",
        type=float,
        )#default=0)
    optimizer.add_argument(
        "--beta2",
        help="Tunes the running average of the squared gradient",
        type=float,
        )#default=0.95)
    optimizer.add_argument(
        "--l2",
        help="Weight decay or l2 regularization",
        type=float,
        )#default=3e-9)
    parser.add_argument(
        "--disable_val_eval",
        help="Disables evaluation on validation data",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--enable_train_eval",
        help="Enables evaluation on training data",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--early_stopping",
        help="Enables early stopping, triggered when no improvements have been"
        + " seen during the past [entered] epochs on validation F1 score",
        type=int,
        )#default=0)
    parser.add_argument(
        "--disable_gradient_clip",
        help="Disable gradient clipping",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--unfactorized",
        help=
        "Treat the lack of label between two words as a regular label. Let the"
        + " labeled subsystem do all the work.",
        type=str2bool,
        )#action="store_true")
    parser.add_argument(
        "--emb_dropout_type",
        type=str,
        choices=["replace", "zero"],
        #default="replace",
        help="Decides how the embedding are dropped" +
        "'replace' replaces the dropped embedding types with a new learnable embedding"
        + " while 'zero' zeros out the embedding and scales the rest")
    parser.add_argument(
        "--load", help="Load trained model", metavar="FILE", )#default="")
    parser.add_argument(
        "--predict_file",
        help="Skips training and instead does predictions on given file." +
        " Predictions are saved in predictions.conllu/predictions.json.",
        metavar="FILE",
        #default="",
        )

    args = parser.parse_args(forced_args)
    d = vars(args)
    dc = cfg_parser.get_args(args.config, args)
    for k, v in dc.items():
        if d[k] is None:
            d[k] = v
        if d[k] is None:
            print("Argument {} not set".format(k))
            print("Restart and set argument {}; I will try to continue for now...".format(k))


    return args


def predict(model, settings, to_predict, elmo, vocabs):
    pred_path = settings.dir + to_predict.split("/")[-1] + ".pred"
    json_path = settings.dir + to_predict.split("/")[-1] + ".json"
    entries, predicted, other_predicted = model.predict(to_predict, elmo)

    json_sentences = []

    with open(pred_path, "w") as fh:
        for sentence in cd.read_col_data(to_predict):
            pred = predicted[sentence.id].numpy()
            if settings.target_style == "scope-":
                cue_matrix = sentence.make_matrix("cues",
                                                  True,
                                                  vocabs[settings.td["cue"]].w2i)
                pred = np.maximum(pred, cue_matrix)
            sentence.update_parse(pred,
                                  settings.target_style,
                                  vocabs[settings.pt].i2w)
            json_sentences.append(cd.convert_col_sent_to_json(sentence))
            fh.write(str(sentence) + "\n")

    with open(json_path, "w") as outfile:
        json.dump(json_sentences, outfile)
    return True


def run_parser(args):
    # For now, assume there always is train, val, and glove data
    if args.seed == -1:
        args.seed = np.random.randint(1234567890)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and not args.force_cpu else "cpu")
    print(device)
    args.device = device
    if torch.cuda.is_available():
        print(torch.cuda.get_device_capability(device))

    args.td = {None: 0, "syn": 1, "sem": 2, "cue": 3, "scope": 4, "scope-": 5}
    args.ot = args.td[args.other_target_style]
    args.pt = args.td[args.target_style]

    args.helpers = None
    if args.help_style:
        args.helpers = [args.td[x] for x in args.help_style.split(",")]

    if not args.dir.endswith("/"):
        args.dir += "/"

    if args.load:
        with open(args.dir + "vocabs.pk", "rb") as fh:
            vocabs = pickle.load(fh)
        #args.vocabs = vocabs
        model = ModelInteractor.factory(args, vocabs)
        model.load(args.load)
    else:
        sentences = cd.read_col_data(args.train)
        if args.vocab is not None:
            with open(args.vocab, "rb") as fh:
                vocabs = pickle.load(fh)
        else:
            _vocabs = make_vocabs(sentences, 0)
            vocabs = Vocabs(*_vocabs)
        with open(args.dir + "vocabs.pk", "wb") as fh:
            pickle.dump(vocabs, fh)
        #args.vocabs = vocabs
        model = ModelInteractor.factory(args, vocabs)

    if args.recycle is not None:
        with open(args.recycle + "vocabs.pk", "rb") as fh:
            other_vocabs = pickle.load(fh)
        with open(args.recycle + "settings.json") as fh:
            other_settings = json.load(fh)
        other_settings = Namespace(**other_settings)
        other_settings.device = args.device
        other = ModelInteractor.factory(other_settings, other_vocabs)
        other.load(args.recycle + "best_model.save")
        model.upd_from_other(other, *args.recycle_layers.split(","))

    if args.freeze is not None:
        model.freeze_params(*args.freeze.split(","))

    if (args.load and args.cont) or args.load is None:
        model.train()

    if (args.load and args.cont and not args.disable_val_eval) or (args.load is None and not args.disable_val_eval):
        # load the best_model.save instead of using the current one
        model = ModelInteractor.factory(args, vocabs)
        model.load(args.dir + "best_model.save")
        predict(model, args, args.val, args.elmo_dev, vocabs)

    if args.predict_file is not None:
        # load the best_model.save instead of using the current one
        model = ModelInteractor.factory(args, vocabs)
        model.load(args.dir + "best_model.save")
        predict(model, args, args.predict_file, args.elmo_test, vocabs)


if __name__ == "__main__":
    run_parser(get_args())
