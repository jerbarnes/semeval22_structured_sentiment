"""
    Relationship gold intensity - polarity prediction
"""
from qualitative_preprocessing import opinion_to_tuple, align_gold_pred
from domain_analysis import open_json
import matplotlib.pyplot as plt
from dataclasses import astuple
from collections import Counter
import pandas as pd
import seaborn as sns
import numpy as np
import copy
import os


def describe_intensity():
    basic_span = "holder, tgt, exp, polarity"
    print("\n##### ANALYSIS INTENSITY: distribution of opinions"\
                " (with match in holder, tgt, exp by > half the teams predictions)"\
                    " with correct and wrong polarity across intensity levels")


def create_intensity_dict(sentences,sentences_keys):
    
    intensity_values = []
    for sent_id in sentences_keys:
        opinions = sentences[sent_id]["opinions"]
        for o in opinions:
            int_val = o["Intensity"]
            if int_val not in intensity_values:
                if str(int_val)=="None":
                    int_val="N0NE"
                intensity_values.append(int_val)

    int_dict = {v:0 for v in intensity_values}

    return int_dict


def plot_intensity(df,setup,corpus):

    outputdir = "../figures/intensity/"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    rows_id = sorted(list(df.index))
    cols_id = df.columns.tolist()
    df.reset_index(drop=True,inplace=True)

    plt.figure(figsize=(5, 5))

    #set width of bars
    barWidth = 0.3
    cols = {}
    for ind,row in df.iterrows():
        cols[ind]=row.values.tolist()
    
    # Set position of bar on X axis
    bars = {}
    for ind,c in enumerate(list(cols.keys())):
        if ind==0:
            br = np.arange(len(cols[c]))
        else:
            br = [x + barWidth for x in bars[ind-1]]
        bars[ind] = br


    colors = ["mediumseagreen","royalblue"]

    for ind,br in enumerate(list(bars.keys())):
        set_color = colors[ind]
        plt.bar(bars[br], cols[br], color=set_color, width = barWidth,
            edgecolor =set_color, label ='first_column')
    
    plt.xticks([r + 0.15 for r in range(len(cols_id))],
        cols_id)
    
    handles = [plt.Rectangle((0,0),1,1, color=colors[ind]) for ind in range(len(rows_id))]
    plt.legend(handles, rows_id)

    output_name = setup+"-"+corpus+"-intensity.pdf"
    plt.savefig(outputdir+output_name)

    plt.clf()


def describe_int_results(setup,results,plot):
    """
    % correct and incorrectly predicted opinions
    wrt intensity level
    """

    for corpus in results.keys():

        frames = []
        c = Counter()
        for d in results[corpus]:
            c.update(d)
    
        for ind,d in enumerate(results[corpus]):
            if ind == 0:
                name = "Correct predictions"
            else:
                name = "Errors"

            df = pd.DataFrame(d, index=[name])
            frames.append(df)
        concat_df = pd.concat(frames)
        increasing_int = ["N0NE","Weak","Average","Slight","Standard","Strong"]
        ordered_cols = [c for c in increasing_int if c in concat_df.columns]
        concat_df = concat_df[ordered_cols]
        print("\n {} : {}".format(setup,corpus))
        print(concat_df)
    
        if plot==True:
            plot_intensity(concat_df,setup,corpus)


def compare_match(g,sent_id,pred_dictionary,easy_dic,difficult_dic):

    hold_gold, tgt_gold, exp_gold, polarity_gold, intensity_gold, txt = g
    match_polarity = 0
    majority_vote = len(pred_dictionary)/2
    for team in pred_dictionary.keys():

        try: 
            p_tpls=opinion_to_tuple(pred_dictionary[team][sent_id])
            matching_pred = align_gold_pred("intensity",g,p_tpls)
        except KeyError:
            # no prediction for that id
            majority_vote-=1

        if len(matching_pred)!=0:
            hold_pred, tgt_pred, exp_pred, polarity_pred, intensity_pred = matching_pred

            if polarity_gold == polarity_pred:
                match_polarity +=1

    if match_polarity > majority_vote:
        easy_dic[intensity_gold]+=1 
    else:
        difficult_dic[intensity_gold]+=1
    
    return easy_dic, difficult_dic


def do_intensity(stp,corpus,gold_keys,gold,pred_dictionary,corpus_results):

    easy_dic = create_intensity_dict(gold,gold_keys)
    difficult_dic = copy.deepcopy(easy_dic)

    for sent_id in sorted(gold_keys):
        if len(gold[sent_id]["opinions"]) == 0:
            continue

        g_tpls=opinion_to_tuple(gold[sent_id])

        for g in g_tpls:
            easy, diff = compare_match(
                g,
                sent_id,
                pred_dictionary,
                easy_dic,
                difficult_dic
                )

    corpus_results[corpus]=[easy_dic,difficult_dic]
    return corpus_results
