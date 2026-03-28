import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="This tool reformats the hickey dataset for use with StrataBionn")

#required arguments
required = parser.add_argument_group("Required arguments")
required.add_argument("-i", "--input", help="Path to input CSV file",required=True)
required.add_argument("-o","--output", help="Output file with Valencia acceptable data")

#reading arguments into parser
args = parser.parse_args()

filename = args.input

data = pd.DataFrame()

def process_single_line(line):
    global data

    if ord(line[-1]) == 10:
        line = line[:-1]
        
    parts = line.split("\t")
    tmp = pd.Series([parts[0]] + parts[2:])
    data = pd.concat([data, tmp], axis=1)

with open(filename, "r") as fp:
    for line in fp.readlines():
        process_single_line(line)

# Rename sampleids

# Get correct column names (currently first row)
data = data.set_axis(data.iloc[0], axis=1)
data = data.drop(0)
data = data.rename(columns={"Taxon_Name": "sampleID"})

data.to_csv(args.output, index=False)