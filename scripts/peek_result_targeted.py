import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_dir(path: str):
    """"""

    dataset = []
    for file in os.listdir(path):
        name = file

        # find _cv.log or _gcv.log
        if name.endswith("_gcv.log"):
            grouped = True
            name = name.replace("_gcv.log", "")
        elif name.endswith("_cv.log"):
            grouped = False
            name = name.replace("_cv.log", "")
        else:
            continue

        # find _no_norm or _norm_XXX
        if name.endswith("_no_norm"):
            normalization = "none"
            name = name.replace("_no_norm", "")
        else:
            idx = name.find("_norm")
            normalization = name[idx + 6 :]
            name = name[:idx]

        # find problem name
        problem = name.replace("_p", "+").replace("_n", "-")

        scores, max_occ, *_ = load_file(os.path.join(path, file))
        dataset.append((problem, normalization, grouped, scores.mean(), max_occ))

    result = pd.DataFrame(
        dataset, columns=["problem", "normalization", "grouped", "score", "max_occ"]
    )

    # sub-plots per problem

    # x-axis: normalization
    # y-axis: score
    # secondary bar: grouping

    n_plots = result["problem"].nunique()
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), sharex=True, sharey=True)

    for problem, ax in zip(result["problem"].unique(), axes.flatten()):
        problem_ds = result[result.problem == problem]

        ax.set_title(problem)
        ax.set_ylabel("score")

        left = problem_ds[problem_ds.grouped == False].sort_values("normalization")
        right = problem_ds[problem_ds.grouped == True].sort_values("normalization")

        df = pd.DataFrame(
            {
                "Grouped CV": list(right.score),
                "CV": list(left.score),
            },
            index=list(left.normalization),
        )

        df.plot.bar(ax=ax)

        ax.plot(list(left.normalization), len(left) * [problem_ds.max_occ], c="black")
        ax.legend(loc="lower right")

    fig.suptitle("Model Assessment")
    fig.tight_layout()
    fig.savefig("tmp.png")


def load_file(path: str):

    with open(path, "r", encoding="utf8") as result_file:
        results = result_file.read()

    max_occ_pattern = re.compile(
        r".*\[root\] \[INFO\] : max rel. occurrence = (0\.\d+)"
    )
    scores_pattern = re.compile(
        r".*\[root\] \[INFO\] : scores: \[(.*)\]"
    )  # recompute mean & std(ddof=1)
    head = re.compile(r".*\[root\] \[INFO\] : feature importances:")
    variable = re.compile(r".*\[root\] \[INFO\] :\s+ (\d+) \(\s*(.*)\) :\s+(\d+\.\d+)")

    max_occ_match = max_occ_pattern.search(results)
    assert max_occ_match is not None
    max_occ = float(max_occ_match.group(1))

    scores_match = scores_pattern.search(results, max_occ_match.pos)
    assert scores_match is not None
    scores = np.fromstring(scores_match.group(1), sep=" ")

    if scores.mean() < max_occ:
        print(f"{scores.mean()=} < {max_occ} : results discarded")
        return scores.mean(), max_occ, None, None

    head_match = head.search(results, scores_match.pos)
    assert head_match is not None

    matches = variable.findall(results, head_match.pos)
    assert len(matches) > 0

    data = [(int(match[0]), match[1], float(match[2])) for match in matches]
    df = pd.DataFrame(data, columns=("id", "name", "importance"))

    # other selection criteria : account for 80% starting with the highest
    df.sort_values(by="importance", inplace=True, ascending=False)
    importance_cumsum = df["importance"].cumsum()
    target = 0.8 * importance_cumsum.iloc[-1]
    limit = importance_cumsum.searchsorted(target, side="right")
    df["cumulate importance"] = importance_cumsum

    return scores, max_occ, df, df.iloc[:limit]
    # print(df.iloc[:limit])


def print_df(df: pd.DataFrame):

    print("\\begin{tabular}{lrl}\n\\toprule\n{}  Name &  MDI Importance  \\\\\n\\midrule\n", end="")
    for row in df.itertuples(index=False):
        print(f"{row[1]} & {row[2]} \\\\\n", end="")
    print("\\bottomrule\n\\end{tabular}\n")
    


if __name__ == "__main__":
    assert len(sys.argv) > 1
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            load_dir(arg)
        else:
            print_df(load_file(arg)[-1])
