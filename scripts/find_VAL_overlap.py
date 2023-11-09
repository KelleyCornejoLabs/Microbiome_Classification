import argparse

import pandas as pd

from nn_classifier import str_norm

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description="This tool splits VALENCIA data into a testing and training split for training")

    required = parser.add_argument_group("Required arguments")
    required.add_argument("-iv", "--input-valencia", help="Path to VALENCUA input CSV file",required=True)
    required.add_argument("-i", "--input", help="Path to input CSV file",required=True)

    # Read arguments into parser
    args = parser.parse_args()

    
    # Read data
    val_data = pd.read_csv(args.input_valencia)

    path = args.input

    # Read data
    test_xlsx = path.endswith(".xlsx")
    test_csv = path.endswith(".csv")

    # Load and validate the input train data
    try:
        # Read appropriate file type
        if test_csv: data = pd.read_csv(path)
        elif test_xlsx: data = pd.read_excel(path)
        else:
            print("Unknown extension")
            exit()
    except FileNotFoundError:
        print("Invalid path")
        exit(1)

    columns = list(map(str_norm, data.columns))
    val_columns = list(map(str_norm, val_data.columns))
    
    same = [c for c in columns if c in val_columns and c not in ['sampleid', 'read_count']]

    print("For use in nn_classifier:")
    print(",".join(same))