# Alexander Symons | make_test_train_split.py | Aug 24 2023

import sys

try:
    import pandas as pd
except:
    print("\nRequired package pandas not available")
    exit()

try:
    import argparse
except:
    print("\nRequired package argparse not available")
    exit()

# Arguments
parser = argparse.ArgumentParser(description="This tool splits VALENCIA data into a testing and training split for training")

required = parser.add_argument_group("Required arguments")
required.add_argument("-i", "--input", help="Path to input CSV file",required=True)
required.add_argument("-o","--output", help="Output file path/prefix")
required.add_argument("-s","--train-split", default=60, help="Percentage of data for training")
required.add_argument("-t","--tolerance", default=0.001, help="Decimal for tolerance between orginal data and outputs for checking")

# Read arguments into parser
args = parser.parse_args()

# Read data
try:
    data = pd.read_csv(args.input)
except FileNotFoundError:
    print("\nInvalid path")
    sys.exit(1)

# For naming consistency with VALENCIA
data = data.rename(columns={'total_reads':'read_count', 'Sample_number_for_SRA':'sampleID'})

# We don't need VALENCIA's predictions and confidence values
data = data.drop(columns=['Val_CST','Val_subCST','I-A_sim','I-B_sim','II_sim','III-A_sim','III-B_sim',
                          'IV-A_sim','IV-B_sim','IV-C0_sim','IV-C1_sim',
                          'IV-C2_sim','IV-C3_sim','IV-C4_sim','V_sim', 'Subject_number', "HC_CST"])

# Found out what proportion of the data each CST makes
entries = data.groupby(['HC_subCST']).count()['sampleID']
total_entries = entries.sum()
prevelance = entries/total_entries
train_count_by_category = (entries * (args.train_split / 100)).astype(int)

# Create each set by putting first x values (as defined in train_count_by_category) 
# in train, and rest in test for each subCST we are controlling for
train_set = pd.DataFrame()
test_set = pd.DataFrame()

subCSTs = list(entries.keys())
for subCST in subCSTs:
    subCST_entries = data.loc[data["HC_subCST"] == subCST]

    subCST_entries = subCST_entries.sample(frac=1).reset_index(drop=True) # Shuffle and return all data with new indicies

    train_set = pd.concat([train_set, subCST_entries[:train_count_by_category[subCST]]])
    test_set = pd.concat([test_set, subCST_entries[train_count_by_category[subCST]:]])

# Check data
train_entries = train_set.groupby(['HC_subCST']).count()['sampleID']
train_total_entries = train_entries.sum()
train_prevelance = train_entries/train_total_entries

test_entries = test_set.groupby(['HC_subCST']).count()['sampleID']
test_total_entries = test_entries.sum()
test_prevelance = test_entries/test_total_entries

tolerance = float(args.tolerance)

if ((abs(train_prevelance - prevelance) > tolerance).any() or (abs(test_prevelance - prevelance) > tolerance).any()):
    print("\nError: problem when splitting data")
    print(f"Prevalance: {prevelance} \n\nTrain: {train_prevelance} \n\nTest: {test_prevelance}")
    sys.exit(2)

# Shuffle SubCST data
train_set = train_set.sample(frac=1).reset_index(drop=True)
test_set = test_set.sample(frac=1).reset_index(drop=True)

# Output to path specified, or out_train.csv
train_path = "out" if args.output == None else args.output
test_path = train_path

train_path += "_train.csv"
test_path += "_test.csv"

train_set.to_csv(train_path, index=False)
test_set.to_csv(test_path, index=False)