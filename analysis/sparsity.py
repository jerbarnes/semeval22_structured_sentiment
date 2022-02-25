from qualitative_preprocessing import opinion_to_tuple, align_gold_pred
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def describe_sparsity(funct):

    match_type = "match"
    if funct == "hte_sparsity":
            print("\n ##### ANALYSIS SPARSITY: how much h, t, e cover in a text")
            print("sparsity of spans to be predicted in easy opinions ")
                
    else:
        print("\n ##### ANALYIS NUM OPINONS")
        print("\navg numb of opinions in texts where easy items come from")
        if "all" in funct:
            match_type="all opinions in text having a match"

    print("(with {} in h,t,e by > half the teams)"\
            " vs. of difficult opinions (with no such match)".format(match_type))


def describe_spars_results(funct, stp, results,plot):

    print("\n"+stp)
    easy = [spar for mylist in list(results.values()) for spar in mylist[0]]
    diff = [spar for mylist in list(results.values()) for spar in mylist[1]]

    print("{} vs. {}".format(round(sum(easy)/len(easy),2),\
        round(sum(diff)/len(diff),2)))


    if funct == "hte_sparsity" and plot==True:
        plot_sparsity(
                    stp,
                    easy,
                    diff)


def label_sparsity(spans,txt):
    """
        How sparse h,t,e spans are in text
    """
    spans_coverage = list(set([x for y in spans for x in y if x!="_"]))
    sparsity = 1 - (len(spans_coverage)/len(txt))
    return sparsity


def plot_sparsity(stp,easy,diff):
    
    outputdir = "../figures/sparsity/"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    cols = ["vals"]
    df_easy, df_diff = pd.DataFrame(easy,columns=cols),\
        pd.DataFrame(diff,columns=cols)
    df_easy["type"], df_diff["type"] = "easy", "difficult"
        
    df = pd.concat([df_easy, df_diff],ignore_index=True)
        
    plt.figure(figsize=(6, 6))
    sns_plot = sns.histplot(df, x="vals", hue="type", element="step")
    
    plt.xlabel("(h,t,e) sparsity", fontsize=10)
    plt.ylabel("Count of sparsity values", fontsize=10)
    new_labels = ['Correct predictions', 'Errors']
    plt.legend(bbox_to_anchor=(0.42, 0.97), title='Labels', labels = new_labels) 
    plt.savefig(outputdir+stp+"-sparsity.pdf")

        
def compare_match(funct,g,sent_id,pred_dictionary,easy, diff):

    hold_gold, tgt_gold, exp_gold, polarity_gold, intensity_gold, txt = g
    majority_vote = len(pred_dictionary)/2

    match_hte = 0

    # check if all teams got h,t,e
    for team in pred_dictionary.keys():
        try: 
            p_tpls=opinion_to_tuple(pred_dictionary[team][sent_id])
            matching_pred = align_gold_pred(funct,g,p_tpls)
        except KeyError:
            majority_vote-=1

        if len(matching_pred)!=0:
            hold_pred, tgt_pred, exp_pred, polarity_pred, intensity_pred = matching_pred
            match_hte+=1

    spars_value = label_sparsity([hold_gold, tgt_gold, exp_gold],txt)
    
    if match_hte> majority_vote:
        match_hte=1
        if funct == "hte_sparsity":
            easy.append(spars_value)
            
    else:
        match_hte=0
        if funct == "hte_sparsity":
            diff.append(spars_value)
    
    return easy, diff, match_hte


def do_sparsity(funct,corpus,gold_keys,gold,pred_dict,results):

    easy, diff = [], []
    for sent_id in sorted(gold_keys):
        if len(gold[sent_id]["opinions"]) == 0:
            continue
        g_tpls=opinion_to_tuple(gold[sent_id])

        matched_opinions=0 
        for g in g_tpls:
            easy, diff, match_hte = compare_match(
                                        funct,
                                        g,
                                        sent_id,
                                        pred_dict,
                                        easy, 
                                        diff
                                        )
            if funct == "opinion_sparsity":
                if match_hte==1:
                    # numb of (h,t,e) in the txt where the current (h,t,e) was recognized
                    easy.append(len(g_tpls))
                else:
                    diff.append(len(g_tpls))
            if "all" in funct and match_hte==1:
                matched_opinions+=1
        
        if "all" in funct:
            # numb of (h,t,e) per text in which all (h,t,e) were recognized
            if matched_opinions==len(g_tpls):
                easy.append(len(g_tpls))
            else:
                diff.append(len(g_tpls))

    results[corpus] = [easy,diff]

    return results