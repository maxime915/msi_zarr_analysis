import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dir(path: str):

    dataset = []
    for file in os.listdir(path):
        name = file

        # find _cv.log or _gcv.log
        if name.endswith("_gcv_cen.log"):
            centroid = True
            name = name.replace("_gcv_cen.log", "")
        elif name.endswith("_gcv.log"):
            centroid = False
            name = name.replace("_gcv.log", "")
        else:
            continue

        # find _no_norm or _norm_XXX
        if name.endswith("_norm_2305"):
            idx = name.find("_norm")
            normalization = name[idx + 6 :]
            name = name[:idx]
        else:
            continue

        # find problem name
        problem = name.replace("_p", "+").replace("_n", "-")

        scores, max_occ = main(os.path.join(path, file))
        dataset.append((problem, centroid, scores.mean(), max_occ))

    result = pd.DataFrame(
        dataset, columns=["problem", "centroid", "score", "max_occ"]
    )
    result.set_index(["problem", "centroid"])
    
    fig, ax = plt.subplots()
    centroid_df = result[result.centroid == True].sort_values("problem")
    non_centroid_df = result[result.centroid == False].sort_values("problem")
    df = pd.DataFrame({
        # "centroid": list(centroid_df.score),
        "non-centroid": list(non_centroid_df.score),
        "max-rel-occ": list(non_centroid_df.max_occ),
    }, index=list(non_centroid_df.problem))
    df.plot.bar(ax=ax)

    ax.set_title("Model Assessment")
    fig.tight_layout()
    fig.savefig("tmp.png")

def main(path: str):
    
    with open(path, 'r', encoding="utf8") as result_file:
        results = result_file.read()

    max_occ_pattern = re.compile(r".*\[root\] \[INFO\] : max rel. occurrence = (0\.\d+)")
    scores_pattern = re.compile(r".*\[root\] \[INFO\] : scores: \[(.*)\]")  # recompute mean & std(ddof=1)
    head = re.compile(r".*\[root\] \[INFO\] : feature importances:")
    variable = re.compile(r".*\[root\] \[INFO\] :\s+ (\d+) \((.*)\) :\s+(\d+\.\d+) \(pm\s+(\d+\.\d+)\)")

    max_occ_match = max_occ_pattern.search(results)
    assert max_occ_match is not None
    max_occ = float(max_occ_match.group(1))
    
    scores_match = scores_pattern.search(results, max_occ_match.pos)
    
    if scores_match is None:
        scores_match = scores_pattern.search(" " * max_occ_match.pos + "[root] [INFO] : scores: [1. 1. 1. 1. 1]", max_occ_match.pos)
    # assert scores_match is not None

    scores = np.fromstring(scores_match.group(1), sep=" ")
    
    if scores.mean() < max_occ:
        print(f"{scores.mean()=} < {max_occ} : results discarded")
        return scores.mean(), max_occ

    head_match = head.search(results, scores_match.pos)
    assert head_match is not None
    
    matches = variable.findall(results, head_match.pos)
    assert len(matches) > 0
    
    data = [(int(match[0]), float(match[1]), float(match[2]), float(match[3])) for match in matches]
    df = pd.DataFrame(data, columns=('id', 'mzs', 'importance', 'standard deviation'))

    path = path.split("/")[-1]
    problem = path[:5] if path[2] == " " else path[:9]

    df.sort_values(by="importance", ascending=False, inplace=True)
    df = df.iloc[:10]
    
    df["Importance"] = (df.importance.map(lambda f: f"{f:.4f} -pm ") + df['standard deviation'].map(lambda f: f"{f:.4f}"))
    df["m/z"] = df.mzs
    
    
    print(df.to_latex(index=False, columns=["m/z", "Importance"], caption=problem))
    
    return scores.mean(), max_occ
    
    # importances = df['importance']
    # q75 = importances.quantile(0.75)
    # q25 = importances.quantile(0.25)
    # q50 = importances.quantile(0.50)
    # outliers = importances > q50 + 1.5 * (q75 - q25)
    
    # print(df[outliers])        

if __name__ == "__main__":
    assert len(sys.argv) > 1
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            load_dir(arg)
        else:
            main(arg)