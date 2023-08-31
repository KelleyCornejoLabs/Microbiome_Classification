# Alexander Symons | Aug 25 2023 | nn_classifier.py
# This file trains and deploys a neural network classifier for microbiome data based on 
# count data for each bacteria
# Inputs: A path to a csv file containing the count data
# Outputs: A file containing parameters for the neural network and predictions for each 
#          sample in the input data

import sys

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

try:
    import numpy as np
except:
    print("Required package numpy not available")
    exit()

try:
    import torch
    import torch.nn as nn
except:
    print("Required package torch not available")
    exit()

parser = argparse.ArgumentParser(description="Trains or uses a neural network classifier on microbiome count data")

# Arguments for tool
arguments = parser.add_argument_group("Arguments")
arguments.add_argument("-i", "--input", help="Path to input data as csv", required=True)
arguments.add_argument("-p", "--path", help="Path to save neural network", default="classifier")
arguments.add_argument("-c","--cuda", help="Boolean to enable/disable cuda. Tries to use gpu by default", default=True)
arguments.add_argument("-a","--accuracy", help="Train until accuracy is with in this tolerance", default=0.01)
arguments.add_argument("-lr","--learning-rate", help="Learning rate for ai", default=0.001)
arguments.add_argument("-me","--max-epochs", help="Maximum number of epochs to train for. None is default", default=None)
arguments.add_argument("-t","--train", help="Should a model be trained based on input data?", default=None)
arguments.add_argument("-s","--split", help="What fraction of data should go to training?", default=0.6)
arguments.add_argument("-m","--metrics-interval", help="How many epochs should training metrics be taken?", default=50)
arguments.add_argument("-l","--loss", help="Loss function. ce (default), nll, or kld", default="ce")
arguments.add_argument("-o","--optim", help="Optimizer. sgd (default), or adam", default="sgd")
arguments.add_argument("-li","--linear", help="Don't use ReLU?", default=False)


# Parse arguments
args = parser.parse_args()

# Set up GPU
device = "cuda" if torch.cuda.is_available() and args.cuda != "False" else "cpu"
if device == "cuda":
    print("Using GPU")

# Load and validate the input data
try:
    df = pd.read_csv(args.input)
except FileNotFoundError:
    print("Could not load input data")
    sys.exit(1)

df_keys = list(df.keys())
if False in [(key in df_keys) for key in ["sampleID", "read_count", "HC_subCST"]]:
    print("Input does not contain sampleID, read_count, or HC_subCST")
    sys.exit(2)

# Shuffle data before splitting
df = df.sample(frac=1).reset_index(drop=True)

# Create helper functions for formatting CST 'labels' for the neural network
all_labels = list(set(df["HC_subCST"]))

def i_to_lbl(i):
    return all_labels[i.argmax()]

def lbl_to_i(lbl):
    return np.eye(len(all_labels))[all_labels.index(lbl)]

# Generate label data
labels = torch.tensor(np.array([lbl_to_i(x) for x in list(df["HC_subCST"])])).to(device).type(torch.float)

# Create normalized the data so each sample is proportional
normalized_data = df.drop(columns=["sampleID", "read_count", "HC_subCST"])
count_columns = normalized_data.columns

for column in count_columns:
    normalized_data[column] /= df["read_count"]

# Format for neural net
training_data = torch.tensor(normalized_data[count_columns].to_numpy()).to(device).type(torch.float)

# Split training data into the train and test sets  
train_split = int(args.split * len(training_data))
print(f"Training split: {train_split}")

X_train, y_train = training_data[:train_split], labels[:train_split]
X_test, y_test = training_data[train_split:], labels[train_split:]

if args.split == 1:
    print("Training on all given data. No data will be reserved for metrics")
    X_test = X_train
    y_test = y_train

print(f"Sizes: train: {len(X_train), len(y_train)}, test: {len(X_test), len(y_test)}")

# Create a Neural Net 
# Current layers do sqrt(in*out) and split in half, and do it again. 51=sqrt(101*13) 101=sqrt(199*51) 26=sqrt(51*13)

if args.linear == "False":
    print("Non-linear")
    classifier = nn.Sequential(
        nn.Linear(in_features=len(training_data[0]), out_features=101),
        nn.ReLU(),
        nn.Linear(in_features=101, out_features=51),
        nn.ReLU(),
        nn.Linear(in_features=51, out_features=26),
        nn.ReLU(),
        nn.Linear(in_features=26, out_features=13),
        nn.Softmax(dim=1)
    ).to(device)
else:
    print("Linear")
    classifier = nn.Sequential(
        nn.Linear(in_features=len(training_data[0]), out_features=101),
        nn.Linear(in_features=101, out_features=51),
        nn.Linear(in_features=51, out_features=26),
        nn.Linear(in_features=26, out_features=13),
        nn.Softmax(dim=1)
    ).to(device)

# TODO: Test following
# Could create list of n layers of nn.Linear() with a formula to calculate layer size (constant or decreasing)
# and then do nn.Sequesntial(*list)
# But that wouldn't be good for consistency across multiple studies

# TODO: try argmax before nll loss?

# TODO: Add column for what each sample was assigned

# Set up optimizer/loss
try:
    lr = float(args.learning_rate)
except TypeError:
    print("Learning rate must be float")
    sys.exit(3)

# NOTE: Cross entropy can take weight for unbalenced sets. try that
# NOTE: Use more metrics (see 2.ipynb). Do per-class accuracy? Confudsion matrix!!
# https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c
losses = {"nll":nn.NLLLoss, "ce":nn.CrossEntropyLoss, "kld":nn.KLDivLoss}
optims = {"sgd": torch.optim.SGD, "adam":torch.optim.Adam}

loss_fn = losses[args.loss]()
optim = optims[args.optim](params=classifier.parameters(), lr=lr)

# Tracking
epoch_count = []
loss_values = []
test_losses = []
test_accuracies = []

try: 
    max_epochs = int(args.max_epochs)
except TypeError:
    print("max epochs must be int")
    sys.exit(3)

try: 
    metrics_interval = int(args.metrics_interval)
except TypeError:
    print("metrics interval must be int")
    sys.exit(3)

try: 
    accuracy = int(args.accuracy)
except TypeError:
    print("accuracy must be int")
    sys.exit(3)

# Define function to find accuracy of
def accuracy_test(lbls, predictions):
    # Do argmax to get index, so as not to do torch.eq on a 2d array
    correct = torch.eq(torch.Tensor([lbl.argmax() for lbl in lbls]), torch.Tensor([p.argmax() for p in predictions])).sum().item()
    return (correct / len(predictions)) * 100

for epoch in range(max_epochs):
    # Forward pass
    y_predictions = classifier(X_train)

    # Calculate loss
    loss = loss_fn(y_predictions, y_train)

    # Backpropagate
    optim.zero_grad()
    loss.backward()
    
    # Gradient Descent
    optim.step()

    # Take metrics
    if epoch % metrics_interval == 0:
        # Evaluation
        classifier.eval()

        with torch.inference_mode():
            test_pred = classifier(X_test)
            test_loss = loss_fn(test_pred, y_test)
            test_accuracy = accuracy_test(y_test, test_pred)

            print(f"Epoch: {epoch} ({((epoch/max_epochs)*100):.2f}%), Loss: {loss}, Test: {test_loss}, Acc: {test_accuracy}")
            torch.save(obj=classifier.state_dict(), f=args.path+"_nn.pt")
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        classifier.train()

        # Stop if accurate enough
        # if test_loss <= accuracy:
        #     break

# Save model
torch.save(obj=classifier.state_dict(), f=args.path+"_nn.pt")

# Write metrics
with torch.inference_mode():
    y_predictions = classifier(X_test)
    accuracy = accuracy_test(y_test, y_predictions)
    
    with open(args.path+"_metrics.txt", "w") as f:
        print(f"Final Accuracy: {accuracy}%")
        f.write(f"Final Accuracy: {accuracy}\n")

        print(f"{len(epoch_count), len(loss_values), len(test_losses)}")
        for i in range(len(epoch_count)):
            f.write(f"Epoch: {epoch_count[i]}, Train loss: {loss_values[i]}, Test Loss: {test_losses[i]}, Accuracy: {test_accuracies[i]}\n")

        f.write("Cases: \n")
        for i in range(10):
            f.write(f"Actual: {i_to_lbl(y_test[i])}, Predicted: {i_to_lbl(y_predictions[i])}\n")