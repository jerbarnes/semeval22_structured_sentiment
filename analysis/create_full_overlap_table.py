import sys
import json
from collections import defaultdict, Counter
team_map = {
    "monolingual": {
        "zhixiaobao": "zhixiaobao",
        "Cong666": "Cong666 (MT-speech)",
        "Team Hitachi ": "Hitachi",
        "colorful": "colorful",
        "Team sixsixsix ": "sixsixsix",
        "KE_AI": "KE\\_AI",
        "Team SeqL ": "SeqL",
        "Team LyS_ACoruña ": "LyS\\_ACoruña",
        "Team ECNU_ICA ": "ECNU\\_ICA",
        "Team ohhhmygosh ": "ohhhmygosh",
    },
    "crosslingual": {
        "Cong666": "Cong666 (MT-speech)",
        "colorful": "colorful",
        "Team Hitachi ": "Hitachi",
        "Team sixsixsix ": "sixsixsix",
        "Team SeqL ": "SeqL",
        "Team ECNU_ICA ": "ECNU\\_ICA",
        "Team Mirs ": "Mirs",
        "Team LyS_ACoruña ": "LyS\\_ACoruña",
        "Team OPI ": "OPI",
        "KE_AI": "KE\\_AI",
    }
}
error_map = {
    "II": "Too late",  # "Late Stop",
    "I": "Too early",  # "Early Stop",
    "III": "Too early",  # "Early Start & Stop",
    "IV": "Too early",  # "Early Start",
    "V": "Other", # "Surround"
    "VI": "Other", # "Contained"
    "VII": "Too late",  # "Late Start",
    "VIII": "Too late",  # "Late Start & Stop",
    "IX": "True negative",  # unnused
    "N": "True positive",  # unnused
    "X": "False positive",
    "XI": "Multiple",
    "XII": "False negative",
}
datasets = {
    "monolingual": [
        "norec",
        "multibooked_ca",
        "multibooked_eu",
        "opener_en",
        "opener_es",
        "mpqa",
        "darmstadt_unis",
    ],
    "crosslingual": [
        "opener_es",
        "multibooked_ca",
        "multibooked_eu",
    ],
}
roles = [
    "Source",
    "Target",
    "Polar_expression",
]
error_labels = [
    "False negative",
    "False positive",
    "Multiple",
    "Other",
    "Too early",
    "Too late",
]

# team -> error type -> dataset -> role -> value
counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

setting = sys.argv[1] if len(sys.argv) > 1 else "monolingual"

with open("assembled_overlap.json") as f:
    for line in f:
        data = json.loads(line)
        if data["mono/single"] != setting:
            continue
        for error_type, number in data["errors"].items():
            counts[data["team"]][error_map[error_type]][data["dataset"]][data["role"]] += number
            counts[data["team"]]["TOTAL"][data["dataset"]][data["role"]] += number

for j, (team, team_fancy) in enumerate(team_map[setting].items()):
    if j:
        # first line of a team, but not of the first team
        print(r"\addlinespace[0.15cm]")
    print(f"\t{team_fancy}", end="")
    for i, error_type in enumerate(error_labels):
        if i:
            # not the first line of a team
            print("\t", end="")
        print(f" & {error_type}", end="")
        for dataset in datasets[setting]:
            for role in roles:
                try:
                    num = counts[team][error_type][dataset][role]
                    num /= counts[team]["TOTAL"][dataset][role]
                    num = f"{num:.0%}"[:-1]
                except ZeroDivisionError:
                    num = "-"
                print(f" & {num}", end="")
        print(r"\\")  # print final newline
