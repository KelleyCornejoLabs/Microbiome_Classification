import argparse

import pandas as pd

from nn_classifier import str_norm

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description="This tool splits VALENCIA data into a testing and training split for training")

    required = parser.add_argument_group("Required arguments")
    required.add_argument("--files", nargs="+", help="List of input CSV files")

    # Read arguments into parser
    args = parser.parse_args()

    files = args.files
    cols = []
    for file in files:
        cols.append(set(map(str_norm, pd.read_csv(file).columns)))

    same = sorted(list(cols[0].intersection(*cols[1:])))
    for to_remove in ['sampleid', 'read_count']:
        if to_remove in same: same.remove(to_remove)

    print("For use in nn_classifier:")
    print(",".join(same))