from qualitative_preprocessing import opinion_to_tuple, align_gold_pred


def decribe_within_graph(funct):

    if funct =="polarity":
        print("\n##### ANALYIS POLARITY: p(correct p | correct (h, t, e))")
        basic_span = "holder, tgt, exp"
        addition = "polarity"
    
    else: 
        if "exact" in funct:
            match_type="exact match"
        else:
            match_type = "weighted match"
        print("\n ##### ANALYIS OPINION SPAN PREDICTIONS: p(exact e | {} (h, t))".
            format(match_type))
        basic_span = "holder, tgt"
        addition = "exp"

    print("\ntot. opinions with match in {} by > half the teams"\
        " vs. percent that also has match in {}".
        format(basic_span, addition))


def describe_results_within_graph(funct, stp, results):
    
    print("\n"+stp)

    ht = 0
    prop = 0

    for corpus in results.keys():
        basic_corpus= list(results[corpus].keys())[0]
        ht+=basic_corpus
        prop+=results[corpus][basic_corpus]

    avg_prop=round((prop/ht),2)
    print("{} vs. {}%".format(ht,avg_prop))


def compare_match(funct,g,sent_id,pred_dictionary,polarity, match_hte, match_ht):

    hold_gold, tgt_gold, exp_gold, polarity_gold, intensity_gold, txt = g

    # teams that correclty predicted
    # ...holder, target, expression and polarity
    match_polarity = 0
    # ...holder, target and exact expression 
    match_exact_exp = 0
    # ...holder, target and expression
    match_hold_tgt_exp = 0
    majority_vote = len(pred_dictionary)/2

    for team in pred_dictionary.keys():
        try: 
            p_tpls=opinion_to_tuple(pred_dictionary[team][sent_id])
            matching_pred = align_gold_pred(funct,g,p_tpls)
        except KeyError:
            # no prediction for that id
            majority_vote-=1

        if len(matching_pred)!=0:
            hold_pred, tgt_pred, exp_pred, polarity_pred, intensity_pred = matching_pred
            
            match_hold_tgt_exp+=1
            if exp_pred==exp_gold:
                match_exact_exp+=1
            
            if polarity_gold == polarity_pred:
                match_polarity +=1
    
    match_pol, match_exact_e, match_hte = 0, 0, 0
    if match_hold_tgt_exp>majority_vote:
        match_hte = 1

        if match_polarity>majority_vote:
            match_pol = 1
    
        if match_exact_exp>majority_vote:
            match_exact_e = 1
    
    return match_pol,  match_exact_e,  match_hte


def do_within_graph(funct,corpus,gold_keys,gold,pred_dict,matching_spans):

    polarity, exact_exp, hold_tgt_exp = 0, 0, 0

    for sent_id in sorted(gold_keys):
        if len(gold[sent_id]["opinions"]) == 0:
            continue

        g_tpls=opinion_to_tuple(gold[sent_id])

        for g in g_tpls:
            match_pol, match_exact_e, \
                match_hte = compare_match(
                                        funct,
                                        g,
                                        sent_id,
                                        pred_dict,
                                        polarity, 
                                        exact_exp, 
                                        hold_tgt_exp
                                        )

            polarity+=match_pol
            exact_exp+=match_exact_e
            hold_tgt_exp+=match_hte

    if funct == "polarity":
        matching_spans[corpus]= {hold_tgt_exp:polarity}

    else: 
        matching_spans[corpus]={hold_tgt_exp:exact_exp}

    return matching_spans
