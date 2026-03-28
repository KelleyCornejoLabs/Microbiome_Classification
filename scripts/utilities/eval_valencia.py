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

from sklearn.metrics import roc_curve
import numpy as np

# Outline args
parser = argparse.ArgumentParser(description="Evaluate the acuracy of VALENCIA on given set")
required = parser.add_argument_group("Required arguments")
required.add_argument("-id", "--input-data", help="Path to file with data that VALENCIA was originally ran on",required=True)
required.add_argument("-ip", "--input-predictions", help="Path to file with data that VALENCIA predicted",required=True)
required.add_argument("-o","--output", help="Output report file prefix", default=None)
required.add_argument("-n","--name", help="Name of classification method",)
required.add_argument("-g","--graph", action=argparse.BooleanOptionalAction, help="Create pop up of confusion matrix", default=True)

# Parse arguments
args = parser.parse_args()

# ROC Curve
def     plot_roc_curve(true_y, y_prob, lbls, method):
    """
    plots the roc curve based of the probabilities
    """

    # Make an ROC curve of each individual feature
    for i in range(len(lbls)):
        fpr, tpr, thresholds = roc_curve(true_y[:,i], y_prob[:,i])
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    plt.title(f"ROC Curve of {method}")
    plt.legend(lbls)

    plt.savefig(f"ROC_{method}")
    if args.output == None: plt.show()

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

print(f"Accuracy: {accuracy:.2f}%")

# Get all lavels, in correct order
all_lbls = list(set(input_data["HC_subCST"]))
all_lbls.sort()

# Create confusion matrix
conf_mat = confusion_matrix(input_data["HC_subCST"], predictions["subCST"])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=all_lbls)
disp.plot()

plt.title(f"Confusion matrix for {args.name}")

print(f"Saving: {args.output}")
plt.savefig(f"{args.output}")
if args.output == None:
    plt.show()

plt.clf()

#I love python
#one_hot = lambda x:np.array([list(np.eye(len(all_lbls))[i if isinstance(i, int) else all_lbls.index(i)]) for i in x])

# Helper function that converts a DataFrame to a list of one-hot encoded lists
def one_hot (x):
    a = []

    # For each sample
    for i in x:
        # Convert string label to an index, and use it to create the one-hot vector
        a.append(list(np.eye(len(all_lbls))[i if isinstance(i, int) else all_lbls.index(i)]))

    print(np.array(a)[:,1])
    return np.array(a)

# Plot the ROC curve
plot_roc_curve(one_hot(input_data["HC_subCST"]), one_hot(predictions["subCST"]), all_lbls, args.name)