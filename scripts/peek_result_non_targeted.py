
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
    variable = re.compile(r".*\[root\] \[INFO\] :\s+ (\d+) \((.*)\) :\s+(\d+\.\d+) \(pm\s+(\d+\.\d+)\)")

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
    
    data = [(int(match[0]), float(match[1]), float(match[2]), float(match[3])) for match in matches]
    df = pd.DataFrame(data, columns=('id', 'mzs', 'importance', 'standard deviation'))
    
    importances = df['importance']
    q75 = importances.quantile(0.75)
    q25 = importances.quantile(0.25)
    q50 = importances.quantile(0.50)
    outliers = importances > q50 + 1.5 * (q75 - q25)
    
    print(df[outliers])        

if __name__ == "__main__":
    assert len(sys.argv) > 1
    for arg in sys.argv[1:]:
        main(arg)