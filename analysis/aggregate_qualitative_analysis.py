from qualitative_preprocessing import corpus_predictions
from domain_analysis import open_json
from within_graph import *
from intensity import *
from sparsity import *

import argparse
import zipfile
import shutil
import glob
import os
import re


def get_args():
    """
    Helper function to get the gold json and predictions json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("predictions")
    args = parser.parse_args()
    
    return args


def unzip_submissions(pred_path):

    subs = [t for t in os.listdir(pred_path) if "submission" in t and t.endswith(".zip")]
    
    for team in subs:
        submission = os.path.join(pred_path+team)
        
        with zipfile.ZipFile(submission,"r") as zip_ref:
            zip_ref.extractall(re.sub(".zip","",submission))


def clean_directory(pred_path):
    """
    Removes unzipped submission files
    """
    pred_files = glob.glob(pred_path+"/*")

    for myf in pred_files:

        if os.path.isdir(myf):
            shutil.rmtree(myf)


def describe_analysis(funct):

    if funct == "polarity" or "holder_target" in funct:
        decribe_within_graph(funct)
    
    elif funct == "intensity":
        describe_intensity()

    else:
        describe_sparsity(funct)


def describe_results(funct, stp, results):

    if funct == "polarity" or "holder_target" in funct:
        describe_results_within_graph(funct, stp, results)
    
    elif funct =="intensity":
        describe_int_results(stp, results,plot=False)
    
    else:
        describe_spars_results(funct, stp, results,plot=False)


def pick_analysis(stp,funct,*args):

    if funct == "polarity" or "holder_target" in funct:
        results = do_within_graph(funct, *args)

    elif funct == "intensity":
        results = do_intensity(stp,*args)

    else:
        results = do_sparsity(funct, *args)

    return results


def analysis(funct):

    describe_analysis(funct)
    for stp in setups:
        results = {}

        for corpus in os.listdir(gold_path):

            if stp == "crosslingual" and corpus not in crosslingual_corpora:
                continue

            goldfile = os.path.join(gold_path, corpus+ "/test_labeled.json")
            gold_keys, gold = open_json(goldfile)
            pred_dict = corpus_predictions(corpus,submission_path,stp)

            results = pick_analysis(
                                    stp,
                                    funct,
                                    corpus,
                                    gold_keys,
                                    gold,
                                    pred_dict,
                                    results)

        describe_results(funct, stp, results)



def main():
    global setups
    global crosslingual_corpora
    setups = [
        "crosslingual",
        "monolingual"
        ]

    crosslingual_corpora = [
        "multibooked_ca",
        "multibooked_eu",
        "opener_es"
    ]

    args = get_args()
    global gold_path
    global submission_path
    gold_path = args.gold
    submission_path = args.predictions

    clean_directory(args.predictions)
    unzip_submissions(args.predictions)

    analysis("holder_target")
    analysis("exact_holder_target")
    analysis("polarity")

    analysis("intensity")
    
    analysis("hte_sparsity")
    analysis("opinion_sparsity")
    analysis("all_opinion_match_sparsity")

    clean_directory(args.predictions)
                


if __name__ == "__main__":
    main()