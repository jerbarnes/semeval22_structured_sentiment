from mpqa_datahelpers import collect_opinion_entities
import numpy as np
import os
import json


def closest_holder(exp_scope, holder_scopes):
    exp_off1 = int(exp_scope.split(":")[0])
    h = np.array([i[0] for i in holder_scopes])
    idx = np.argmin(np.abs(h - exp_off1))
    return holder_scopes[idx]

def get_all_holder_ids(holder):
    holder_ids = holder.split(",")
    for i in range(len(holder_ids)):
        for j in range(len(holder_ids)):
            if i < j:
                h = "{0},{1}".format(holder_ids[i], holder_ids[j])
                holder_ids.append(h)
    return holder_ids

def get_opinions(lre_file, doc_file):
    text = open(doc_file).read()
    agents, attitudes, attitudes_type, targets = collect_opinion_entities(lre_file)

    new = {}

    s = doc_file.split("/")
    idx = "/".join((s[-2], s[-1]))

    new["sent_id"] = idx
    new["text"] = text

    opinions = []

    for i, line in enumerate(open(lre_file)):
        line = line.strip()
        line_tab = line.split("\t")
        if i < 5:
            continue
        if line_tab[3] == "GATE_direct-subjective" and len(line_tab) > 4:
            exp_scope = line_tab[1].replace(",", ":")
            off1, off2 = exp_scope.split(":")
            off1 = int(off1)
            off2 = int(off2)
            exp_tokens = text[off1:off2]
            expression = [exp_tokens, exp_scope]
            #print("Expression: " + str(expression))

            arguments = line_tab[4]

            # get the polarity
            if len(arguments.split("polarity=")) > 1:
                ds_polarity = arguments.split("polarity=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                #print("Polarity: " + str(ds_polarity))

                # get the intensity
                if len(arguments.split("expression-intensity=")) > 1:
                    expression_intensity = arguments.split("expression-intensity=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not expression_intensity:
                        expression_intensity = None
                    #print("intensity: " + expression_intensity)

                # get the holder
                if len(arguments.split("nested-source=")) > 1:
                    holder = arguments.split("nested-source=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not holder:
                        holder = None
                    else:
                        holder_ids = get_all_holder_ids(holder)
                        holder_scopes = [agents[i][0] for i in holder_ids if len(agents[i]) > 0]
                        closest = closest_holder(exp_scope, holder_scopes)
                        holder_scope = "{0}:{1}".format(*closest)
                        holder_tokens = text[closest[0]:closest[1]]
                        if holder_tokens == "":
                            holder = None
                        else:
                            holder = [holder_tokens, holder_scope]
                    #print("holder: " + str(holder))

                # keep only those with sentiment attitudes
                if len(arguments.split("attitude-link=")) > 1:

                    # attitude-link="a4, a6" ---> a4,a6
                    attitude_ids = arguments.split("attitude-link=")[1].split('"')[1].replace(' ', '').split(',')

                    for aid, att_id in enumerate(attitude_ids):
                        att_type = attitudes_type[att_id] if attitudes_type[att_id] else 'none'

                        target_ids = attitudes[att_id]

                        for target_id in target_ids:
                            target_spans = targets[target_id]
                            target_tokens = ""
                            target_span_str = ""
                            for span in target_spans:
                                off1 = int(span[0])
                                off2 = int(span[1])
                                target_tokens += text[off1:off2]
                                target_span_str += "{0}:{1};".format(off1, off2)

                            target = [target_tokens, target_span_str[:-1]]
                            #print("type: " + att_type)
                            #print("target: " + str(target))

                if "sentiment" in att_type:
                    opinion = {"Source": holder,
                               "Target": target,
                               "Polar_expression": expression,
                               "Polarity": ds_polarity,
                               "Intensity": expression_intensity}

                    opinions.append(opinion)
                    #print(att_type)
                    #print(opinion)

                #print()

    new["opinions"] = opinions

    return new

if __name__ == "__main__":

    train = [l.strip() for l in open("datasplit/filelist_train0").readlines()]
    data = [("train", train)]

    for name, fnames in data:

        processed = []

        for fname in fnames:
            lre_file = "database.mpqa.2.0/man_anns/{0}/gateman.mpqa.lre.2.0".format(fname)
            doc_file = "database.mpqa.2.0/docs/{0}".format(fname)
            try:
                new = get_opinions(lre_file, doc_file)
                if len(new["opinions"]) > 0:
                    processed.append(new)
            except:
                pass

        with open(os.path.join("{0}.json".format(name)), "w") as out:
            json.dump(processed, out)
