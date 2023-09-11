# Alexander Symons | Sept 11 2023 | eval_valencia.py
# This script takes the VALENCIA output file and compares it to the file it was originally given,
# to produce an accuracy score for the dataset that can be compared to other models

try:
    import pandas as pd
except:
    print("Required package pandas not available")
    exit()

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except:
    print("Required package sklearn not available")
    exit()

try:
    import matplotlib.pyplot as plt
except:
    print("Required package sklearn not available")
    exit()

try:
    import argparse
except:
    print("Required package argparse not available")
    exit()

# Outline args
parser = argparse.ArgumentParser(description="Evaluate the acuracy of VALENCIA on given set")
required = parser.add_argument_group("Required arguments")
required.add_argument("-id", "--input-data", help="Path to file with data that VALENCIA was originally ran on",required=True)
required.add_argument("-ip", "--input-predictions", help="Path to file with data that VALENCIA predicted",required=True)
required.add_argument("-o","--output", help="Output report file prefix", default=None)

# Parse arguments
args = parser.parse_args()

# Read data
input_data = pd.read_csv(args.input_data)
predictions = pd.read_csv(args.input_predictions)

# Make sure they contain same samples in same order for comparison
input_data = input_data.sort_values("sampleID")
predictions = predictions.sort_values("sampleID")

if False in list(input_data["sampleID"] == predictions["sampleID"]):
    print("Input data and input predictions do not contain same samples")
    exit()

# Calculate accuracy
correct = list(input_data["HC_subCST"] == predictions["subCST"])
accuracy = (correct.count(True) / len(correct)) * 100

print(f"Valencia accuracy: {accuracy:.2f}%")

all_lbls = list(set(input_data["HC_subCST"]))
all_lbls.sort()

conf_mat = confusion_matrix(input_data["HC_subCST"], predictions["subCST"])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=all_lbls)
disp.plot()

if args.output == None:
    plt.show()
else:
    plt.savefig(f"{args.output}")