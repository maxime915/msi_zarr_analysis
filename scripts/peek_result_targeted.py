
import re
import sys
import numpy as np
import pandas as pd

def main(path: str):
    
    with open(path, 'r', encoding="utf8") as result_file:
        results = result_file.read()

    max_occ_pattern = re.compile(r".*\[root\] \[INFO\] : max rel. occurrence = (0\.\d+)")
    scores_pattern = re.compile(r".*\[root\] \[INFO\] : scores: \[(.*)\]")  # recompute mean & std(ddof=1)
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
        return

    head_match = head.search(results, scores_match.pos)
    assert head_match is not None
    
    matches = variable.findall(results, head_match.pos)
    assert len(matches) > 0
    
    data = [(int(match[0]), match[1], float(match[2])) for match in matches]
    df = pd.DataFrame(data, columns=('id', 'name', 'importance'))
    
    # other selection criteria : account for 80% starting with the highest
    df.sort_values(by='importance', inplace=True, ascending=False)
    importance_cumsum = df['importance'].cumsum()
    target = 0.8 * importance_cumsum.iloc[-1]
    limit = importance_cumsum.searchsorted(target, side='right')
    df['cumulate importance'] = importance_cumsum
    print(df.iloc[:limit])


if __name__ == "__main__":
    assert len(sys.argv) > 1
    for arg in sys.argv[1:]:
        main(arg)