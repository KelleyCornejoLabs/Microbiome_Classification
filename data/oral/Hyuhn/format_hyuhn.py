import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="This tool creates reformats the Hyuhn data to be acceptable by VALENCIA/StrataBionn")

#required arguments
required = parser.add_argument_group("Required arguments")
required.add_argument("-i", "--input", help="Path to input CSV file",required=True)
required.add_argument("-o","--output", help="Output file with Valencia acceptable data")

#reading arguments into parser
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t").T

df["sampleID"] = df.index
df = df.drop("taxID")
df["read_count"] = df.sum(numeric_only=True, axis=1)

cols = list(df.columns)
cols.remove("sampleID")
cols.remove("read_count")

dfr = df.reindex(columns=["sampleID", "read_count"]+cols)

dfr = dfr.rename({c:c.lstrip() for c in cols}, axis=1)

dfr.to_csv(args.output, index=False)