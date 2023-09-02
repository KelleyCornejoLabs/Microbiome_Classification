# Alexander Symons | Aug 25 2023 | nn_classifier.py
# This file trains and deploys a neural network classifier for microbiome data based on 
# count data for each bacteria
# Inputs: A path to a csv file containing the count data
# Outputs: A file containing parameters for the neural network and predictions for each 
#          sample in the input data

import sys
import time

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

# Set up GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")

losses = {"nll":nn.NLLLoss, "ce":nn.CrossEntropyLoss, "kld":nn.KLDivLoss}
optims = {"sgd": torch.optim.SGD, "adam":torch.optim.Adam}

def load_data(train_path, test_path):
    # Load and validate the input data
    try:
        dftr = pd.read_csv(train_path)
    except FileNotFoundError:
        print("Could not load input training data")
        sys.exit(1)

    try:
        dfte = pd.read_csv(test_path)
    except FileNotFoundError:
        print("Could not load input test data")
        sys.exit(1)

    def check_keys(df_keys, in_type):
        if False in [(key in df_keys) for key in ["sampleID", "read_count", "HC_subCST"]]:
            print(f"{in_type} input does not contain sampleID, read_count, or HC_subCST")
            sys.exit(2)

    check_keys(list(dftr.keys()), "Training")
    check_keys(list(dfte.keys()), "Testing")

    # Shuffle data before splitting
    dftr = dftr.sample(frac=1).reset_index(drop=True)
    dfte = dfte.sample(frac=1).reset_index(drop=True)

    # Create helper functions for formatting CST 'labels' for the neural network
    if set(dftr["HC_subCST"]) != set(dfte["HC_subCST"]):
        print("Training and test set do not contain same CSTs")
        sys.exit(2)

    all_labels = list(set(dftr["HC_subCST"]))

    def i_to_lbl(i):
        return all_labels[i.argmax()]

    def lbl_to_i(lbl):
        return np.eye(len(all_labels))[all_labels.index(lbl)]

    # Generate label data
    test_labels = torch.tensor(np.array([lbl_to_i(x) for x in list(dfte["HC_subCST"])])).to(device).type(torch.float)
    train_labels = torch.tensor(np.array([lbl_to_i(x) for x in list(dftr["HC_subCST"])])).to(device).type(torch.float)

    # Create normalized the data so each sample is proportional
    normalized_train_data = dftr.drop(columns=["sampleID", "read_count", "HC_subCST"])
    normalized_test_data = dfte.drop(columns=["sampleID", "read_count", "HC_subCST"])

    if len([(i,j) for i, j in zip(normalized_test_data.columns, normalized_train_data.columns) if i != j ]) != 0:
        print("Training and test set do not contain same count columns, or they are not in the same order")
        print(f"Found: {len([(i,j) for i, j in zip(normalized_test_data.columns, normalized_train_data.columns) if i != j ])}")
        sys.exit(2)

    count_columns = normalized_train_data.columns

    # TODO: Handle if two files have same columns in different order?

    for column in count_columns:
        normalized_train_data[column] /= dftr["read_count"]
        normalized_test_data[column] /= dfte["read_count"]

    # Format for neural net
    training_data = torch.tensor(normalized_train_data[count_columns].to_numpy()).to(device).type(torch.float)
    testing_data = torch.tensor(normalized_test_data[count_columns].to_numpy()).to(device).type(torch.float)

    # Split training data into the train and test sets  
    print(f"Training split: {len(training_data)}, Testing split: {len(testing_data)}")

    X_train, y_train = training_data, train_labels
    X_test, y_test = testing_data, test_labels

    # if split:
    #     print("Training on all given data. No data will be reserved for metrics")
    #     X_test = X_train
    #     y_test = y_train

    print(f"Sizes: train: {len(X_train), len(y_train)}, test: {len(X_test), len(y_test)}")

    # Found out what proportion of the data each CST makes
    entries = dftr.groupby(['HC_subCST']).count()['sampleID']

    # TODO: Test this with more metrics. It will affect how the model learns rarer classes
    # Get in order of what index each label is in training data
    ordered_prevelence = torch.tensor([1/entries[all_labels[i]] for i in range(len(all_labels))]).to(device)
    ordered_prevelence *= 1/ordered_prevelence.min()

    return X_train, y_train, X_test, y_test, all_labels, ordered_prevelence

# Create a Neural Net 
# Current layers do sqrt(in*out) and split in half, and do it again. 51=sqrt(101*13) 101=sqrt(199*51) 26=sqrt(51*13)
def generate_model(linear, train_features, hidden_features, classes):
    if not linear:
        print(f"Non-linear {train_features} -> {classes}")
        classifier = nn.Sequential(
            nn.Linear(in_features=train_features, out_features=101),
            nn.ReLU(),
            nn.Linear(in_features=101, out_features=51),
            nn.ReLU(),
            nn.Linear(in_features=51, out_features=26),
            nn.ReLU(),
            nn.Linear(in_features=26, out_features=classes),
            nn.Softmax(dim=1)
        ).to(device)
    else:
        print(f"Linear {train_features} -> {classes}")
        classifier = nn.Sequential(
            nn.Linear(in_features=train_features, out_features=101),
            nn.Linear(in_features=101, out_features=51),
            nn.Linear(in_features=51, out_features=26),
            nn.Linear(in_features=26, out_features=13),
            nn.Softmax(dim=1)
        ).to(device)

    return classifier

# TODO: Test nmodel architectures

# Define function to find accuracy of model
def accuracy_test(lbls, predictions):
    # Do argmax to get index, so as not to do torch.eq on a 2d array
    correct = torch.eq(torch.Tensor([lbl.argmax() for lbl in lbls]), torch.Tensor([p.argmax() for p in predictions])).sum().item()
    return (correct / len(predictions)) * 100

def train(classifier, X_train, y_train, X_test, y_test, lr, max_epochs, metrics_interval, 
          accuracy, loss_type, optim_type, linear, all_labels, ordered_prevalence, path):

    def i_to_lbl(i):
        return all_labels[i.argmax()]
    
    def lbl_to_i(lbl):
        return np.eye(len(all_labels))[all_labels.index(lbl)]
    
    # TODO: try argmax before nll loss?

    # TODO: Add column for what each sample was assigned in output

    # Set up optimizer/loss

    # NOTE: Use more metrics (see 2.ipynb). Do per-class accuracy? Confudsion matrix!!
    # https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c

    if loss_type == "ce":
        loss_fn = losses[loss_type](weight=ordered_prevalence)
    else:
        loss_fn = losses[loss_type]()
    optim = optims[optim_type](params=classifier.parameters(), lr=lr)

    # Tracking
    epoch_count = []
    loss_values = []
    test_losses = []
    test_accuracies = []

    # Track time
    start_time = time.time()

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
                torch.save(obj=classifier.state_dict(), f=path+"_nn.pt")
                epoch_count.append(epoch)
                loss_values.append(loss)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)

            classifier.train()

            # Stop if accurate enough
            # if test_loss <= accuracy:
            #     break

    # Save model
    torch.save(obj=classifier.state_dict(), f=path+"_nn.pt")

    # Calculate time
    time_now = time.time()
    time_taken = f"{int((time_now - start_time) / 3600)}:{int(((time_now - start_time) / 60) % 60)}:{int((time_now - start_time) % 60)}"

    # Write metrics
    with torch.inference_mode():
        y_predictions = classifier(X_test)
        accuracy = accuracy_test(y_test, y_predictions)

        with open(path+"_metrics.txt", "w") as f:
            print(f"Final Accuracy: {accuracy}% in {time_taken}")
            f.write(f"Final Accuracy: {accuracy}% in {time_taken}\n")

            print(f"Test Config: lr: {lr}, linear: {not (linear == 'False')}, loss_fn: {loss_type}, optim: {optim_type}\n")
            f.write(f"Test Config: lr: {lr}, linear: {not (linear == 'False')}, loss_fn: {loss_type}, optim: {optim_type}\n")

            print(f"{len(epoch_count), len(loss_values), len(test_losses)}")
            for i in range(len(epoch_count)):
                f.write(f"Epoch: {epoch_count[i]}, Train loss: {loss_values[i]}, Test Loss: {test_losses[i]}, Accuracy: {test_accuracies[i]}\n")

            f.write("Cases: \n")
            for i in range(10):
                f.write(f"Actual: {i_to_lbl(y_test[i])}, Predicted: {i_to_lbl(y_predictions[i])}\n")

        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains or uses a neural network classifier on microbiome count data")

    # Arguments for tool
    arguments = parser.add_argument_group("Arguments")
    arguments.add_argument("-itr", "--input-train", help="Path to train input data as csv", required=True)
    arguments.add_argument("-ite", "--input-test", help="Path to test input data as csv", required=True)
    arguments.add_argument("-p", "--path", help="Path to save neural network", default="classifier")
    arguments.add_argument("-a","--accuracy", help="Train until accuracy is with in this tolerance", default=0.01)
    arguments.add_argument("-lr","--learning-rate", help="Learning rate for ai", default=0.001)
    arguments.add_argument("-me","--max-epochs", help="Maximum number of epochs to train for. None is default", default=None)
    arguments.add_argument("-t","--train", help="Should a model be trained based on input data?", default=None)
    arguments.add_argument("-s","--split", help="What fraction of data should go to training?", default=0.6)
    arguments.add_argument("-m","--metrics-interval", help="How many epochs should training metrics be taken?", default=50)
    arguments.add_argument("-l","--loss", help="Loss function. ce (default), nll, or kld", default="ce")
    arguments.add_argument("-o","--optim", help="Optimizer. sgd (default), or adam", default="sgd")
    arguments.add_argument("-li","--linear", help="Don't use ReLU?", default=False)
    arguments.add_argument("-sd","--seed", help="Seed rng", default=None)

    # Parse arguments
    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        try:
            seed = int(args.seed)
        except TypeError:
            print("Seed must be an int")
            sys.exit(1)

        torch.manual_seed(seed)

    linear = args.linear == "True"

    X_train, y_train, X_test, y_test, all_labels, ordered_prevelence = load_data(args.input_train, args.input_test)
    classifier = generate_model(linear, len(X_train[0]), 0, len(y_train[0]))

    try:
        lr = float(args.learning_rate)
    except TypeError:
        print("Learning rate must be float")
        sys.exit(3)

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

    train(classifier, X_train, y_train, X_test, y_test, lr, max_epochs, metrics_interval, accuracy, 
          args.loss, args.optim, linear, all_labels, ordered_prevelence, args.path)
