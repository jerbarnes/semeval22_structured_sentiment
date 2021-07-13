import codecs
from collections import defaultdict
import json


def collect_opinion_entities(filename):
    agents = defaultdict(list)
    attitudes = defaultdict(list)
    attitudes_type = defaultdict(list)
    targets = defaultdict(list)

    ####################################
    # GET AGENTS (OPINION HOLDERS)
    ####################################

    file_lre = codecs.open(filename, "r", encoding="utf-8").readlines()
    # construct an agents dictionary: agents[agent] = (start_agent, end_agent)
    for (i, line) in enumerate(file_lre):
        line = line.strip()
        line_tab = line.split("\t")
        if i < 5:
            continue

        if line_tab[3] == "GATE_agent" and len(line_tab) >= 5:
            try:
                agent = line_tab[4].split("nested-source=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
            except IndexError:
                # 1 4578,4581   string  GATE_agent   agent-uncertain="somewhat-uncertain"
                try:
                    agent = line_tab[4].split("id=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                except IndexError:
                    pass

            if agent:
                agents[agent].append([int(x) for x in line_tab[1].split(',')])


    #######################################
    # GET ATTITUDES (OPINION EXPRESSIONS)
    #######################################

    for i, line in enumerate(file_lre):
        if i < 5:
            continue  # first 5 lines are meta-data

        line_tab = line.split("\t")
        if line_tab[3] == "GATE_attitude" and len(line_tab[4].split(' ')) > 1:
            if len(line_tab[4].split("target-link=")) > 1:
                target_ids = line_tab[4].split("target-link=")[1].split("\n")[0].split('"')[1].replace(' ', '').split(',')
                if not target_ids:
                    target_ids = ['none']
            else:
                target_ids = ['none']

            if len(line_tab[4].split("attitude-type=")) > 1:
                type_full = line_tab[4].split("attitude-type=")[1].split('"')[1]
            else:
                type_full = 'none'


            if len(line_tab[4].split("id=")) > 1:
                attitude_id = line_tab[4].split("id=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                attitudes[attitude_id].extend([target_id for target_id in target_ids])
                attitudes_type[attitude_id] = type_full

    #####################################
    # GET TARGETS (OPINION TARGETS)
    #####################################

    # construct a target dictionary: targets[target_id] = (start_target, end_target)
    for (i, line) in enumerate(file_lre):
        if i < 5:
            continue
        line_tab = line.split("\t")

        if line_tab[3] == "GATE_target" and len(line_tab) > 4 and len(line_tab[4].split("id=")) > 1:
            target = line_tab[4].split("id=")[1].split("\n")[0].split('"')[1].replace(', ', ',')

            if len(line_tab[4].split("target-uncertain=")) > 1:
                target_uncertain = line_tab[4].split("target-uncertain=")[1].split('"')[1]
                if not target_uncertain:
                    target_uncertain = 'unk'
            else:
                target_uncertain = 'no'
            targets[target].append([int(x) for x in line_tab[1].split(',')])

    return agents, attitudes, attitudes_type, targets
