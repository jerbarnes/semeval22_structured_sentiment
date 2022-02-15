import json
from collections import defaultdict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

with open("relevant_teams") as f:
    relevant_teams = {line.rstrip("\n") for line in f}

error_map = {
    "II": "Too late",  # "Late Stop",
    "I": "Too early",  # "Early Stop",
    "III": "Too early",  # "Early Start & Stop",
    "IV": "Too early",  # "Early Start",
    "V": "Surround",
    "VI": "Contained",
    "VII": "Too late",  # "Late Start",
    "VIII": "Too late",  # "Late Start & Stop",
    "X": "False positive",
    "XI": "Multiple",
    "XII": "False negative",
}
order = ["False positive", "False negative", "Multiple", "Too early", "Too late", "Surround", "Contained"]

data = defaultdict(list)
with open("assembled_overlap.json") as f:
    for line in f:
        line = json.loads(line)
        if line["team"] not in relevant_teams:
            continue
        total = sum(line["errors"].values())
        for error_type in line["errors"]:
            if error_type in ("IX", "N"):  # not real errors
                continue
            value = line["errors"][error_type] / total
            if value > 0.025 or True:
                data[line["dataset"], line["mono/single"]].append(
                    {
                        "Team": line["team"],
                        "Error type": error_map[error_type],
                        "Role": line["role"],
                        "Dataset": line["dataset"],
                        "Relative frequency": value,
                        "Absolute counts": line["errors"][error_type],
                    }
                )

sns.set(font_scale=0.7)
for dataset, monomulti in data:
    print("Scattering", dataset)
    df = pd.DataFrame(data[dataset, monomulti])
    axis = sns.scatterplot(
        data=df, x="Team", y="Relative frequency", hue="Role", style="Error type"
    )
    axis.tick_params(axis="x", rotation=30)
    lgd = axis.legend(loc="center right", bbox_to_anchor=(1.3, 0.6), fancybox=True)
    plt.tight_layout()
    plt.autoscale()
    fig = axis.get_figure()
    fig.savefig(
        f"scatter_{dataset}_{monomulti}.pdf",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.clf()

    if monomulti == "monolingual":
        print("Boxing", dataset)
        axis = sns.boxplot(data=df, x="Error type", y="Relative frequency", hue="Role", order=order)
        axis.tick_params(axis="x", rotation=15)
        fig = axis.get_figure()
        fig.savefig(f"box_{dataset}_{monomulti}.pdf")
    plt.clf()

for setting in ("monolingual", "crosslingual"):
    print("Boxing", setting)
    df = pd.DataFrame(item for (_ds, monomulti), items in data.items() for item in items if monomulti == setting)
    axis = sns.boxplot(data=df, x="Error type", y="Relative frequency", hue="Role", order=order)
    axis.tick_params(axis="x", rotation=15)
    fig = axis.get_figure()
    fig.savefig(f"box_all_{setting}.pdf")
    plt.clf()
