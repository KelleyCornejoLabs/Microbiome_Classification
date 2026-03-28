# Alexander Symons | Sep 18 2024 | dataset_stats.py

# Time os and math are part of the standard library
import time
import os
import math

try:
    import pandas as pd
except:
    print("Required package pandas not available")
    exit()

try:
    import argparse
except:
    print("Required package argparse not available")
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains or uses a neural network classifier on microbiome count data")

    # Arguments for tool
    arguments = parser.add_argument_group("Arguments")
    arguments.add_argument("-ds", "--data-set", help="Path to dataset to check", default=None)

    # Parse arguments
    args = parser.parse_args()

    path = args.data_set
    if path == None:
        print("Data path required")
        exit()

    # Find data format
    fmt_xlsx = path.endswith(".xlsx")
    fmt_csv = path.endswith(".csv")

    # Load and validate the input data
    try:
        # Read appropriate file type
        if fmt_csv: df = pd.read_csv(path)
        elif fmt_xlsx: df = pd.read_excel(path)
        else:
            # Print error and exit if unknown type
            print(f"ERR: Unkown file format for data {path}")
            print("Known file formats are .xlsx and .csv")
            exit()
    except FileNotFoundError:
        # Error and exit if couldn't load
        print("ERR: Could not load input data")
        exit()

    sample_id = "sampleID" in df.columns
    hc_subcst = "HC_subCST" in df.columns
    read_count = "read_count" in df.columns

    print("Valencia formatting fields found:")
    print("sampleID:", "FOUND" if sample_id else "MISSING (REQUIRED)")
    print("hc_subcst:", "FOUND" if sample_id else "MISSING (UNLABELED)")
    print("read_coubt:", "FOUND" if sample_id else "MISSING (OPTIONAL)")

    species = set(df.columns) - {"sampleID", "HC_subCST", "read_count"}

    print()
    print("Columns:", ", ".join(species))

    if hc_subcst:
        csts = sorted(list(set(df["HC_subCST"])))

        print()
        print("All CSTs:", ", ".join(csts))

        counts = df["HC_subCST"].value_counts()

        for cst in csts:
            print(f"{cst}:", counts[cst])