import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="This tool creates reformats the Manghi data to be acceptable by VALENCIA/StrataBionn")

#required arguments
required = parser.add_argument_group("Required arguments")
required.add_argument("-i", "--input", help="Path to input CSV file",required=True)
required.add_argument("-o","--output", default="output.csv", help="Output file with Valencia acceptable data")

#reading arguments into parser
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t")
df = df.rename(columns={"Unnamed: 0":"sampleID", "diagnosis_binary":"HC_subCST"})
df = df.drop(columns=["family_id", "role"])

df.to_csv(args.output, index=False)