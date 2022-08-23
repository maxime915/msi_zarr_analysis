"search in combined.log"

import json
import re
from collections import Counter

log_path = "logs/classification_task/combined.log"
with open(log_path, "r", encoding="utf8") as log_file:
    log_data= log_file.read()

header_pat = re.compile(r".*checking for class imbalance:")
limit_pat = re.compile(r".*max rel. occurrence = (\d\.\d+)")
score_pat = re.compile(r".* : scores: \[.*\] \((\d\.\d+) pm.*")
model_pat = re.compile(r".* : model: (.*)")

counter = Counter()

pos = 0
iteration = -1
while True:
    iteration += 1

    header_match = header_pat.search(log_data, pos)
    if not header_match:
        break

    limit_match = limit_pat.search(log_data, header_match.pos)
    assert limit_match
    
    limit_val = float(limit_match.group(1))
    
    score_match = score_pat.search(log_data, limit_match.pos)
    assert score_match
    
    score_val = float(score_match.group(1))
    
    # must be successful, otherwise irrelevant
    if score_val < limit_val:
        pos = score_match.end(0)
        continue
    
    model_match = model_pat.search(log_data, score_match.pos)
    assert model_match
    
    model_val = model_match.group(1)
    pos = model_match.end(0)

    # only add grouped CV
    counter[model_val] += 1


pretty_class_dict = {
    "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>": "ET",
    "<class 'sklearn.ensemble._forest.RandomForestClassifier'>": "RF",
    "<class 'sklearn.tree._classes.DecisionTreeClassifier'>": "DTC",
}

print("""\\begin{table}[ht]
\\centering
\\begin{tabular}{lcccc}
\\toprule
{}  Model &  max\\_depth & max\\_features & n\\_estimators & count \\\\
\\midrule""")

for model_param, count in counter.most_common():
    idx = model_param.find(",")
    model, params = model_param[:idx], model_param[idx+14:]
    model = pretty_class_dict[model]

    params = params.replace("'", "\"").replace("None", "null")
    params = json.loads(params)

    max_depth = params.get("max_depth")
    max_features = params.get("max_features", " ")
    n_estimators = params.get("n_estimators", " ")

    # awesome ! now print as a latex table

    print(f"{model} & {max_depth} & {max_features} & {n_estimators} & {count} \\\\")



print("""\\bottomrule
\\end{tabular}
    \\caption{}
    \\label{}
\\end{table}
""")

r"""\begin{table}[ht]
\centering
\begin{tabular}{lcccc}
\toprule
{}  Model &  max\_depth & max\_features & n\_estimators & count \\
\midrule
ET & 20 & sqrt & 1000 & 6 \\
ET & None & sqrt & 200 & 5 \\
ET & 20 & None & 200 & 4 \\
ET & 20 & sqrt & 200 & 3 \\
RF & None & sqrt & 200 & 3 \\
RF & 20 & sqrt & 1000 & 3 \\
ET & 20 & None & 500 & 2 \\
RF & 20 & sqrt & 500 & 2 \\
ET & 20 & sqrt & 500 & 2 \\
DTC & 1 &   &   & 2 \\
RF & 20 & None & 1000 & 1 \\
DTC & 20 &   &   & 1 \\
ET & 20 & None & 1000 & 1 \\
DTC & None &   &   & 1 \\
RF & 20 & None & 200 & 1 \\
RF & None & sqrt & 500 & 1 \\
ET & None & None & 200 & 1 \\
RF & None & None & 1000 & 1 \\
\bottomrule
\end{tabular}
    \caption{}
    \label{}
\end{table}
"""