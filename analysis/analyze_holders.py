import json
import pathlib
from collections import Counter, defaultdict

COUNTS = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
BASE = pathlib.Path("../data")
ROLES = {
    "Source": "Holder",
    "Target": "Target",
    "Polar_expression": "Expression",
}

# add entries here to map from technical dataset names to "fancy" names for TeX
DS_MAP = {}

for dataset in BASE.glob("*"):
    if not dataset.is_dir():
        continue
    for part, file in zip(("train", "test"), ("train.json", "test_labeled.json")):
        file = (dataset / file)
        if not file.exists():
            break
        with file.open() as f:
            data = json.load(f)
        for row in data:
            for opinion in row["opinions"]:
                for role in ROLES:
                    COUNTS[dataset.name][role][part][bool(opinion[role][0])] += 1

# OUTPUT TABLE:
print(r"""\begin{tabular}{lrrrrrr}
    \toprule
    """, end="")
for role in ROLES.values():
    print(r" & \multicolumn{2}{c}{", role, "}", end="")
print(r"\\")
print(r"    \cmidrule{lr}{2-4}\cmidrule{lr}{5-7}\cmidrule{lr}{8-10}")
print("    Dataset", *["& train & test"]*3, r"\\")
print(r"    \midrule")
for dataset in COUNTS:
    print("   ", DS_MAP.get(dataset, dataset.replace("_", r"\_")), end="")
    for role in ROLES:
        for part in ("train", "test"):
            true = COUNTS[dataset][role][part][True]
            false = COUNTS[dataset][role][part][False]
            print(f" & {true/(true + false):.2%}".replace("%", r"\%"), end="")
    print(r"\\")
print(r"""    \bottomrule
\end{tabular}""")
