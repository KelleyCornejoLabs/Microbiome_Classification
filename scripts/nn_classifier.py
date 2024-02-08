# Alexander Symons | Aug 25 2023 | nn_classifier.py
# This file trains and deploys a neural network classifier for microbiome data based on 
# count data for each bacteria
# Inputs: A path to a csv file containing the count data
# Outputs: A file containing parameters for the neural network and predictions for each 
#          sample in the input data

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

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
    skl = True
except:
    print("Optional package sklearn not available")
    skl = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from matplotlib.colors import ListedColormap
    mpl = True
except:
    print("Optional package matplotlib not available")
    mpl = False

try:
    import conorm
    cnrm = True
except:
    print("Optional package conorm not available. TMM normalization cannot be used")
    cnrm = False

# Set up GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")

losses = {"nll":nn.NLLLoss, "ce":nn.CrossEntropyLoss, "kld":nn.KLDivLoss}
optims = {"sgd": torch.optim.SGD, "adam":torch.optim.Adam}

# Put in alphabetical order so two datasets with same columns in different order works
def reorder(df):
    return df.reindex(sorted(df.columns), axis=1)

# Normalize the data so its all the same
def str_norm(x: str):
    prefixes = ["g_", "o_", "k_", "f_", "d_", "c_"]

    x = x.lower().replace(' ', '_')

    # Remove any prefix
    if x[1] == '_':
        x = x[2:]

    # Return lower_camel_case
    return x

# Load one file to tensor
def load_file(path: str, expect_labeled: bool, drop: None|list[str] = [], 
              keep: None|list[str] = None, debug: bool = False, 
              norm: str = "none", regex_remove: list[str] = [''],) -> \
              tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Loads classified or unclassified data. Dropping columns in drop, only keeping columns 
    in keep, or removing columns according to a regex (if they are supplied). Will only return
    data type specified, removing labels from labeled data if unlabeled data is requested. 
    Returns (data, count_columns) if unlabeled data is requested, and (data, labels, all_labels,
    ordered_prevelence, count_columns) if labeled data requested."""

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

    # Chek if data is labeled
    labeled = "HC_subCST" in list(df.keys())
    if debug: print(f"DBG: Data is {' not' if not labeled else ''}labeled")

    # Other required keys
    required_keys = ["sampleID", "read_count"]
    
    # Exit if wrong type of data found
    if expect_labeled != labeled:
        # Exit if expected labels not found
        if expect_labeled == True:
            print(f"ERR: Expected labeled data, found unlabeled")
            exit(1)

        # If unexpected labels, ignore them
        print("WARN: Labels found but unlabeled data requested. Ignoring labels")
        labeled = False

        # Add HC_subCST to the keys that will be dropped
        required_keys.append("HC_subCST")


    # Check for required keys
    for key in required_keys:
        if not key in list(df.keys()):
            print(f"ERR: Requred key {key} not found in data.")
            exit(1)

    # Drop columns for simplified model
    ignore = False
    if drop != []:
        for col in drop:
            try:
                df = df.drop(columns=[col])
            except KeyError:
                print(f"WARN: Failed to drop {col} from dataset")

    # If this is labeled data, 
    if labeled:
        # Get the superset of all possible classifications
        all_labels = sorted(list(set(df["HC_subCST"])))

        # Create function to convert string names to a one-hot vector
        lbl_to_i = lambda lbl: np.eye(len(all_labels))[all_labels.index(lbl)]
        
        # Make np array of one-hot encoded vectors, convert to cuda(?) tensors
        labels = np.array([lbl_to_i(x) for x in list(df["HC_subCST"])])
        labels = torch.tensor(labels).to(device).type(torch.float)

        # Found out what proportion of the data each CST makes
        entries = df.groupby(["HC_subCST"]).count()["sampleID"]

        # TODO: Test this with more metrics. It will affect how the model learns rarer classes <---- Look more into IV-C4
        # TODO: Try different measures of prevelance. Giving rarer classes such a bigger importance will
        # impact the accuracy of more common classes a lot

        # Return normalized pervelance stats for the data, for weighting SGD
        ordered_prevelence = torch.tensor([1/entries[all_labels[i]] for i in range(len(all_labels))]).to(device)
        ordered_prevelence *= 1/ordered_prevelence.min()

        # Add HC_subCST to the keys that will be dropped
        required_keys.append("HC_subCST")

    # Drop the required keys, assign new df to normalized variable
    normalized_data = df.drop(columns=required_keys)

    # Get columns that contain count data
    count_columns = normalized_data.columns

    # If columns are explicitly provided to be kept
    if keep != None:
        # Normalize all column names to be kept
        keep = list(map(str_norm, keep))

        # Check each column
        for col in count_columns:
            if not str_norm(col) in keep:
                if debug: print(f"DBG: Dropping: {col}")
                # If we find a string that shouldn't be kept, attempt to drop
                try: 
                    normalized_data = normalized_data.drop(columns=[col])
                except KeyError:
                    print(f"ERR: Column {col} not found to drop")
                    exit(1)


    # Allow removal of elements by regex. If regexes provided,
    if regex_remove != ['']:
        # Attempt to remove all of them
        for regex in regex_remove:
            # Use regex to select columns
            to_remove = list(normalized_data.filter(regex=regex))
            if debug: print(f"DBG: Removing from regex {regex}: {', '.join(to_remove)}")

            # Only allow columns not selected
            normalized_data = normalized_data[normalized_data.columns.drop(to_remove)]

    # Get columns that contain count data which remain after pruning
    count_columns = normalized_data.columns

    # Normalize the columns based on count data to adjust for sample 'quality'
    for column in count_columns:
        normalized_data[column] /= df["read_count"]

    # Perform the requested normalizations
    if norm == "log":
        for column in count_columns:
            normalized_data[column] /= list(map(lambda x:math.log10(x+0.001), normalized_data[column]))

    elif norm == "tmm":
        if not cnrm:
            print("ERR: Conorm package required for TMM normalization")
            exit(1)

        normalized_data = conorm.tmm(normalized_data.T).T

    # Order consistently
    normalized_data = reorder(normalized_data)

    # Format for neural net as cuda(?) float tensor
    data = torch.tensor(normalized_data.to_numpy()).type(torch.float).to(device)

    # Get updated count columns in correct order
    count_columns = normalized_data.columns

    # Return data and information about it
    if labeled: return data, labels, all_labels, ordered_prevelence, count_columns
    else: return data, count_columns

# Load data for training and validation from paths to csv test and training data files
def load_data(train_path: str, test_path: str, drop: None|list[str] = [], 
              keep : None|list[str] = None, debug:bool = False, norm:str = "none",
              regex_remove:list[str] = ['']) -> tuple[torch.Tensor, torch.Tensor, 
              torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Returns: X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, count_columns.\n
    Loads data from supplied (str) paths to csv training and testing data. Returns
    normalized 'x' input data as a tensor on the device, 'y' output data as one hot tensors
    corresponding to that classes index in all_labels, a sorted list encoding the order of the classes,
    ordered_prevelence which is a tensor on the device with adjused prevelences of each class, and 
    the column names of those containing input data. Drop drops the rows provided as strigns in a list, 
    Keep keeps the rows provided as a list of strings if they exist."""

    # Load training data
    X_train, y_train, all_labels_train, ordered_prevelence, count_columns_train = \
                load_file(train_path, True, drop, keep, debug, norm, regex_remove)
    
    # Load testing data
    X_test, y_test, all_labels_test, _, count_columns_test = \
                load_file(test_path, True, drop, keep, debug, norm, regex_remove)
    
    # Function to find differences in count coluns/labels
    def find_different (train, test):
        return [(i,j) for i, j in zip(train, test) if i != j]

    # Detect if cound columns are different after loading/processing, warn user if they are
    different_cols = find_different(count_columns_train, count_columns_test)
    if len(different_cols) != 0:
        print("ERR: Training and test set do not contain same count columns, or they are not in the same order")
        print(f"Found {len(different_cols)} (after pre-processing): {''.join(different_cols)}")
        exit(2)

    # Detect if cound columns are different after loading/processing, warn user if they are
    different_lbls = find_different(all_labels_train, all_labels_test)
    if len(different_lbls) != 0:
        print("ERR: Training and test set do not contain same labels, or they are not in the same order")
        print(f"Found {len(different_lbls)} (after pre-processing): {''.join(different_lbls)}")
        exit(2)

   
    # Print debug info for user
    if debug: print(f"DBG: Training split: {len(X_train)}, Testing split: {len(X_test)}")
    if debug: print(f"DBG: Sizes: train: {len(X_train), len(y_train)}, test: {len(X_test), len(y_test)}")
    if debug: print(f"DBG: First 5 labels of train: {', '.join(map(lambda i:all_labels_train[i.argmax()], y_train[:5]))}")

    return X_train, y_train, X_test, y_test, all_labels_train, ordered_prevelence, count_columns_train

# Load unlabeled data for classification by model
def load_unlabeled(path: str, drop: list[str]|None = [], keep: list[str]|None = None, 
                   debug:bool = False) -> tuple[torch.Tensor, list[str]]:
    """Load data without labels to be fed to model for classification. A list of column names
    to explicity keep (drop everything else) or provide a list of column names to drop"""

    # TODO: forward normalization and regex

    # load_unlabeled is kind of obsolete now...
    return load_file(path, False, drop, keep, debug)
    

# Create a Neural Net 
# Current layers do sqrt(in*out) and split in half, and do it again. 51=sqrt(101*13) 101=sqrt(199*51) 26=sqrt(51*13)
def generate_model(linear: bool, train_features: int, hidden_features: int, classes: int, dbg: bool, 
                   old:bool = False, droprate = 0.3) -> tuple[nn.Sequential, str, None]:
    """Generates a fresh model based on model linearity, number of training features, number of 
    hidden features, and number of output features. Returns the model, a string representation
    of its structure, and None (for optimizer)"""

    if old:
        # Old architecture is WAY overcomplicated, not reccomended
        if not linear:
            structure = f"OLD Non-linear {train_features} -> {hidden_features} -> {classes}"
            if dbg: print(structure)
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
            structure = f"OLD Linear {train_features} -> {hidden_features} -> {classes}"
            if dbg: print(structure)
            classifier = nn.Sequential(
                nn.Linear(in_features=train_features, out_features=101),
                nn.Linear(in_features=101, out_features=51),
                nn.Linear(in_features=51, out_features=26),
                nn.Linear(in_features=26, out_features=classes),
                nn.Softmax(dim=1)
            ).to(device)
    else:
        if not linear:
            structure = f"NEW Non-linear {train_features} -> {hidden_features} -> {classes}"
            if dbg: print(structure)
            classifier = nn.Sequential(
                nn.Linear(in_features=train_features, out_features=hidden_features),
                nn.LeakyReLU(),
                nn.Dropout(p=droprate),
                nn.Linear(in_features=hidden_features, out_features=classes),
                nn.Softmax(dim=1)
            ).to(device)
        else:
            structure = f"NEW Linear {train_features} -> {hidden_features} -> {classes}"
            if dbg: print(structure)
            classifier = nn.Sequential(
                nn.Linear(in_features=train_features, out_features=hidden_features),
                nn.Linear(in_features=hidden_features, out_features=classes),
                nn.Softmax(dim=1)
            ).to(device)

    # Model, structure:str, optimizer (None)
    return classifier, structure, None

# Define function to find accuracy of model
def accuracy_test(lbls: torch.Tensor, predictions: torch.Tensor) -> float:
    """Returns a float accuracy given a one-hot list of labels, and a one-hot list of predictions"""

    # Do argmax to get index, so as not to do torch.eq on a 2d array
    correct = torch.eq(torch.Tensor([lbl.argmax() for lbl in lbls]), torch.Tensor([p.argmax() for p in predictions])).sum().item()
    return (correct / len(predictions)) * 100

# Returns a list of how many times model's first guess was right, second guess, ...
def top_n_accuracy(model: nn.Sequential, data: torch.Tensor, lbls: torch.Tensor) -> list[float]:
    """Takes a model, data, and labels for the data and returns a list of how many times the choice
        at that index was correct. [times_choice_1_was_correct, times_choice_2_was_correct, ...]"""
    
    # Make predictions
    model.eval()
    with torch.inference_mode():
        predictions = model(data)

    # Make predictions list of lists, and convert lbls from one-hot tensors
    predictions = list(predictions.cpu().numpy())
    predictions = list(map(lambda x:list(x), predictions))
    lbls = list(lbls.argmax(dim=1).cpu().numpy())

    # Find how accuracies for each guess
    accuracies = []
    for i in range(len(predictions[0])):
        # Find which class was the model's ith choice (1st choice, 2nd choice...)
        ith_choice = list(map(lambda x:x.index(sorted(x, reverse=True)[i]), predictions))
        # Append the number of times it was correct
        accuracies.append([ith_choice[j] == lbls[j] for j in range(len(ith_choice))].count(True))

    return accuracies

# Evaluate feature importance using the model. Return dict of each feature's importance
def feature_importance(model: nn.Sequential, data: torch.Tensor, lbls: torch.Tensor, 
                       features: list[str]) -> dict[str, float]:
    """Given a model, data, and labels, return a dictionary containing the importance 
    ( |accuracy - accuracy_with_feature_permuted| ) of each feature"""

    # NOTE: works well for increasing accuracy, but not so well for less common classes
    # Should weight less frequent classes more heavily

    with torch.inference_mode():
        test_predictions = model(data)

    # Baseline accuracy
    standard_score = accuracy_test(lbls, test_predictions)
    
    importances = {feat:0 for feat in features}

    for i,feat in enumerate(features):
        # Create copy of data so it isnt overwritten
        data_cpy = data.detach().clone()
        
        # Permute selected feature
        feature = data_cpy[:,i].cpu().numpy()
        permuted = torch.Tensor(np.random.permutation(feature)).to(device)
        data_cpy[:,i] = permuted

        # Test accuracy on permuted data
        with torch.inference_mode():
            test_predictions = model(data_cpy)

        permuted_score = accuracy_test(lbls, test_predictions)

        # Find difference between scores and save it
        importances[feat] = standard_score - permuted_score

    sorted_importances = dict(sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True))
    return sorted_importances

# Trains the model and take metrics
def train(classifier: nn.Sequential, X_train: torch.Tensor, y_train: torch.Tensor, 
          X_test: torch.Tensor, y_test: torch.Tensor, lr: float, max_epochs: int, 
          metrics_interval: int, thresh: float, loss_type: str, optim_type: str, 
          linear: bool, all_labels: list[str], ordered_prevalence: torch.Tensor, 
          path: str, structure: str, keys: list[str], optim: torch.optim.Optimizer|None = None, 
          debug: bool = False, patience: int = 50) -> float:
    """Train model on given data with given hyperparameters and return max accuracy. Latest version of model may not be 
    best peforming on testing data, so load the model after training to get it"""

    def i_to_lbl(i):
        return all_labels[i.argmax()]
    
    def lbl_to_i(lbl):
        return np.eye(len(all_labels))[all_labels.index(lbl)]

    # TODO: Add column for what each sample was assigned in output

    # Set up optimizer/loss

    if loss_type == "ce":
        loss_fn = losses[loss_type](weight=ordered_prevalence)
    else:
        loss_fn = losses[loss_type]()

    if optim == None:
        if optim_type == "sgd":
            optim = optims[optim_type](params=classifier.parameters(), lr=lr, momentum=0.9)
        else:
            optim = optims[optim_type](params=classifier.parameters(), lr=lr)

    # NOTE: Use a different lr_scheduler for SGD
    #if optim_type != "sgd":
    if patience != 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.2, patience=patience, verbose=debug)

    # NLLLoss requires scalars, not one-hot vectors
    if loss_type == "nll":
        y_train = y_train.argmax(dim=1)

    # Tracking
    epoch_count = []
    loss_values = []
    test_losses = []
    test_accuracies = []

    # Track time
    start_time = time.time()

    # Only save best model
    max_acc = 0

    for epoch in range(max_epochs):
        # Forward pass
        y_predictions = classifier(X_train)

        # Calculate loss
        if loss_type == "nll":
            y_predictions = y_predictions.log_softmax(dim=1)

        loss = loss_fn(y_predictions, y_train)

        # Backpropagate
        optim.zero_grad()
        loss.backward()

        # Gradient Descent
        optim.step()

        # Step lr scheduler
        #if optim_type != "sgd":
        if patience != 0:
            scheduler.step(loss)

        # Take metrics
        if epoch % metrics_interval == 0:
            # Evaluation
            classifier.eval()

            with torch.inference_mode():
                test_pred = classifier(X_test)
                if loss_type == "nll":
                    test_loss = loss_fn(test_pred, y_test.argmax(dim=1))
                else:
                    test_loss = loss_fn(test_pred, y_test)

                test_accuracy = accuracy_test(y_test, test_pred)

                if debug:
                    print(f"Epoch: {epoch} ({((epoch/max_epochs)*100):.2f}%), Loss: {loss}, ", end="")
                    print(f"Test: {test_loss}, Acc: {test_accuracy}, lr:{optim.state_dict()['param_groups'][0]['lr']}")

                # Save if new best
                if test_accuracy > max_acc:
                    max_acc = test_accuracy
                    torch.save({"model": classifier.state_dict(),
                                "structue": structure,
                                "features": keys,
                                "all_labels": all_labels,
                                "optim_type": optim_type,
                                "lr": lr,
                                "optim": optim.state_dict()}, f=path + "_nn.pt")
                    
                epoch_count.append(epoch)
                loss_values.append(loss)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)

            classifier.train()

            # TODO: Doesnt work for SGD, also accuracy is incorrect term now
            # Stop if accurate enough
            if optim.state_dict()["param_groups"][0]["lr"] <= thresh:
                break

    # Save model
    #torch.save(obj=classifier.state_dict(), f=path+"_nn.pt")
    
    # Calculate time
    time_now = time.time()
    time_taken = f"{int((time_now - start_time) / 3600)}:{int(((time_now - start_time) / 60) % 60)}:{int((time_now - start_time) % 60)}"

    # Write metrics
    with torch.inference_mode():
        test_pred = classifier(X_test)
        with open(path + "_metrics.txt", "w") as f:
            if debug: print(f"Final Max Accuracy: {max_acc}% in {time_taken}")
            f.write(f"Final Max Accuracy: {max_acc}% in {time_taken}\n")

            if debug: print(f"Test Config: lr: {lr}, linear: {linear}, loss_fn: {loss_type}, optim: {optim_type}\n")
            f.write(f"Test Config: lr: {lr}, linear: {linear}, loss_fn: {loss_type}, optim: {optim_type}\n")

            for i in range(len(epoch_count)):
                f.write(f"Epoch: {epoch_count[i]}, Train loss: {loss_values[i]}, Test Loss: {test_losses[i]}, Accuracy: {test_accuracies[i]}\n")

            f.write("Cases: \n")
            for i in range(10):
                f.write(f"Actual: {i_to_lbl(y_test[i])}, Predicted: {i_to_lbl(test_pred[i])}\n")
        
        # If matplot imported, save graph
        if mpl:
            prep = lambda x:list(map(lambda y:y.cpu().item(), x))

            plt.plot(epoch_count, prep(loss_values))
            plt.plot(epoch_count, prep(test_losses))
            #plt.plot(epoch_count, test_accuracies)
            plt.legend(["Train Loss", "Test Loss", "Test Accuracy"])
            plt.savefig(f"{path}_plt.png")

    return max_acc

# Pepare data. Convert from one-hot cuda tensor to scalar np array
def prep_data(data: torch.Tensor) -> np.ndarray:
    """Converts data from one-hot cuda tensor to np array of floats Scikit"""
    prepped = data.cpu()
    prepped = prepped.argmax(dim=1)
    return prepped.numpy()

# Test model, taking various metrics
def test(model: nn.Sequential, X_test: torch.Tensor, y_test: torch.Tensor, 
         all_labels: list[str]) -> None:
    """Test model, print accuracy and confusion matrix"""

    if not skl:
        print("Scikit Learn required for testing")
        return

    # Get predictions
    model.eval()
    with torch.inference_mode():
        y_predictions = model(X_test)

    predictions = prep_data(y_predictions)
    lbls = prep_data(y_test)

    # Take metrics
    # Accuracy
    print(f"accuracy: {accuracy_test(y_test, y_predictions):.2f}%")

    print('  '.join(all_labels))

    # Confusion matrix
    conf_mat = confusion_matrix(lbls, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=all_labels)
    disp.plot()
    plt.show()
    print(conf_mat)

    # F1
    f1 = f1_score(lbls, predictions, average="weighted")
    print(f"F1 (weighted): {f1:.4f}")

colors = ListedColormap([(a[0]/255, a[1]/255, a[2]/255, a[3]) for a in [(45, 31, 125, 1),
    (91, 142, 197, 1),
    (125, 197, 236, 1),
    (59, 160, 142, 1),
    (19, 108, 45, 1),
    (143, 142, 45, 1),
    (217, 187, 108, 1),
    (91, 19, 0, 1),
    (198, 91, 108, 1),
    (161, 59, 91, 1),
    (125, 31, 76, 1),
    (160, 59, 142, 1),
    (220, 20, 60, 1)]])

# Store all data relating to data visualization and associated callbacks
class Plotter:
    feature_1 = 0
    feature_2 = 0

    def __init__(self, ax, fig, X_test, y_test, keys, labels):
        self.ax = ax
        self.fig = fig
        self.X_test = X_test
        self.y_test = y_test
        self.keys = keys
        self.labels = labels
        self.first_time = True

    # Draw fresh scatterplot with the current features
    def update_scatter(self):

        self.ax.clear()

        s = self.ax.scatter(self.X_test[:,self.feature_1], self.X_test[:,self.feature_2], c=self.y_test, cmap=colors)
        
        if self.first_time:
            self.fig.legend(s.legend_elements()[0], self.labels,
                            loc="center right", title="CST")
            self.first_time = False


        self.ax.set_xlabel(self.keys[self.feature_1])
        self.ax.set_ylabel(self.keys[self.feature_2])
        self.ax.set_title(f"Features: {self.keys[self.feature_1]} x {self.keys[self.feature_2]}")
        
        self.fig.canvas.draw()

    # Select feature1/2 to plot and compare
    def next_f1(self, event):
        if self.feature_1 < len(self.X_test[0]) - 1: self.feature_1 += 1
        self.update_scatter()

    def prev_f1(self, event):
        if self.feature_1 > -len(self.X_test[0]): self.feature_1 -= 1
        self.update_scatter()

    def next_f2(self, event):
        if self.feature_2 < len(self.X_test[0]) - 1: self.feature_2 += 1
        self.update_scatter()

    def prev_f2(self, event):
        if self.feature_2 > -len(self.X_test[0]): self.feature_2 -= 1
        self.update_scatter()

# Plot boundary line of two features
def plot_correlations(model: nn.Sequential, X_test: torch.Tensor, y_test: torch.Tensor, 
                      all_labels: list[str], keys: list[str]) -> None:
    """Plot the predicted class on 2d plot with two chosen features as x and y axis"""

    if not plt:
        print("Matplot required for plot_correlations.")
        return

    # Get data to explore
    model.eval()
    with torch.inference_mode():
        predictions = model(X_test)
        predictions = prep_data(predictions)

    # Make sure its in right format in CPU memory
    y_test = prep_data(y_test)
    X_test = X_test.cpu().numpy()
    
    # Create subplot instance
    fig, ax = plt.subplots(figsize=(7,6))

    plotter = Plotter(ax, fig, X_test, y_test, keys, all_labels) # Track plotting variables
    plotter.update_scatter()

    # Add buttons to the subplot
    plt.subplots_adjust(bottom=0.2)
    next_f1_ax = plt.axes([0.75, 0.05, 0.15, 0.07])
    prev_f1_ax = plt.axes([0.58, 0.05, 0.15, 0.07])
    next_f2_ax = plt.axes([0.41, 0.05, 0.15, 0.07])
    prev_f2_ax = plt.axes([0.24, 0.05, 0.15, 0.07])

    button_next_f1 = Button(next_f1_ax, "Next feature1", color="green", hovercolor="blue")
    button_prev_f1 = Button(prev_f1_ax, "Previous featue1", color="green", hovercolor="blue")
    button_next_f2 = Button(next_f2_ax, "Next feature2", color="green", hovercolor="blue")
    button_prev_f2 = Button(prev_f2_ax, "Previous feature2", color="green", hovercolor="blue")
    
    # Register click event callbacks to the plotter's handler functions
    button_next_f1.on_clicked(plotter.next_f1)
    button_prev_f1.on_clicked(plotter.prev_f1)
    button_next_f2.on_clicked(plotter.next_f2)
    button_prev_f2.on_clicked(plotter.prev_f2)    

    plt.show()


# Load model at path from the path_nn.pt
def load_model(path: str, keys: None|list[str] = None, return_features: bool = False, 
               debug: bool = False) -> tuple[nn.Sequential, str, torch.optim.Optimizer, 
                                             list[str]|None, list[str]|None]:
    """Load a model and return info about it. Returns [classifier, structure, optim
    features, classes]."""

    # Load 'checkpoint'
    try:
        checkpoint = torch.load(path + "_nn.pt")
    except:
        print(f"Couldn't read {path}_nn.pt")
        exit(1)

    # Validate architecture type, linearity, and layer sizes
    structure = checkpoint["structue"].split(" ")
    
    old_arch = True if structure[0] == "OLD" else False if structure[0] == "NEW" else None
    linear = True if structure[1] == "Linear" else False if structure[1] == "Non-linear" else None

    if old_arch == None or linear == None:
        print(f"Erroneous structure data in {path}_nn.pt; Old: {old_arch}, Linear: {linear}")
        exit(1)

    try:
        input_size = int(structure[2])
        hidden_size = int(structure[4])
        output_size = int(structure[6])
    except TypeError:
        print(f"Erroneous structure data in {path}_nn.pt; layers must be int -> int -> int")
        exit(1)

    if keys != None and set(keys) != set(checkpoint["features"]):
        print(f"Features used in model and features provided do not match.")
        exit(1)

    # Create the classifier and load its state from nn.pt
    classifier, structure, _ = generate_model(linear, input_size, hidden_size, output_size, debug, old=old_arch)
    classifier.load_state_dict(checkpoint["model"])

    # Load optimizer
    if checkpoint.get("optim_type") != None:
        optim = optims[checkpoint["optim_type"]](params=classifier.parameters(), lr=checkpoint["lr"])
        optim.load_state_dict(checkpoint["optim"])
    else:
        # This is for the case where the model has been converted from an older version of the program
        optim = None

    if checkpoint.get("features") is not None:
        features = list(checkpoint["features"])
    else:
        features = None

    if checkpoint.get("all_labels") is not None:
        all_labels = list(checkpoint["all_labels"])
    else:
        all_labels = None

    # Model, structure (str), optimizer, features (optional)
    if return_features: return classifier, structure, optim, features, all_labels
    else: return classifier, structure, optim


# Load model and return all the info about it
def get_model_info(path: str) -> tuple[dict, str, list[str], str, float, dict]:
    """Returns the model state dictionary, model structure, list of output classes, the 
    features the model uses, optimizer type used in training, the learning rate from 
    when it was saved, and the optimizer's state dictionary"""

    # Load 'checkpoint' dict
    try:
        checkpoint = torch.load(path + "_nn.pt")
    except:
        print(f"Couldn't read {path}_nn.pt")
        exit(1)

    # Get all the info from the model checkpoint if it exists
    model_sd = checkpoint.get("model")
    structure = checkpoint.get("structue")
    all_labels = checkpoint.get("all_labels")
    features = checkpoint.get("features")
    optim_type = checkpoint.get("optim_type")
    lr = checkpoint.get("lr")
    optim_sd = checkpoint.get("optim")
    
    return model_sd, structure, all_labels, features, optim_type, lr, optim_sd

# Train a simpler model based on only the columns with importance surpassing threshold
def train_simpler_model(train_path: str, test_path: str, sorted_importances: dict[str, float], 
                        imporatance_threshold: float, lr: float, max_epochs: int, metrics_interval: int, 
                        thresh: float, loss_type: str, optim_type: str, linear: bool, path: str, focus: list[str], 
                        norm:str = "none", debug: bool = False, hidden: None|int = None, patience: int = 100, 
                        models: int = 1, regex_remove:list[str] = [","]) -> tuple[nn.Sequential, torch.Tensor, 
                        torch.Tensor, list[str]]:
    """Train a simpler model using the given settings and return the best model. Columns used are 
    the entries in sorted_importances are greater than importance_threashold"""

    # Determine columns to cut
    if focus == None:
        if debug: print("DBG: Determining columns to ignore manually")
        unimportant_cols = [key for key, value in sorted_importances.items() if value < imporatance_threshold]
        
        # Determine new data containing the Significant columns
        SX_train, Sy_train, SX_test, Sy_test, Sall_labels, Sordered_prevelence, Skeys = load_data(train_path, test_path, 
                                                                                                  drop=unimportant_cols,
                                                                                                  debug=debug, norm=norm,
                                                                                                  regex_remove=regex_remove)
    else: # Or manually keep columns
        if debug: print("DBG: Focusing on provided columns")
        SX_train, Sy_train, SX_test, Sy_test, Sall_labels, Sordered_prevelence, Skeys = load_data(train_path, test_path, 
                                                                                                  keep=focus, debug=debug, 
                                                                                                  norm=norm, 
                                                                                                  regex_remove=regex_remove)

   
    if debug: print(f"Training simpler model on columns: {','.join(list(Skeys))}")

    if hidden == None:
        hidden = int(round(len(SX_train[0]) * (2/3) + len(Sy_train[0])))

        # Hidden layers shouldn't be bigger than classes or outputs. If it is, pick bigger of two
        hidden = max(min(hidden, round(len(SX_train[0]))), round(len(Sy_train[0])))

    # Get most accurate model
    accuracies = []
    for i in range(models):
        # Simplified classifier creation and training
        Sclassifier, Sstructure, Soptim = generate_model(linear, len(SX_train[0]), hidden, len(Sy_train[0]), debug)

        acc = train(Sclassifier, SX_train, Sy_train, SX_test, Sy_test, lr, max_epochs, metrics_interval, thresh, 
              loss_type, optim_type, linear, Sall_labels, Sordered_prevelence, f"{path}_simplified_{i}", Sstructure, 
              Skeys, patience=patience, debug=debug)
        
        accuracies.append(acc)

    # Get rid of the worse models
    best_model_index = accuracies.index(max(accuracies))
    rename_best(f"{path}_simplified", best_model_index, models)

    # Latest might not be best performer (which is what gets saved)
    Sclassifier, _, _ = load_model(f"{path}_simplified", debug=debug)
    return Sclassifier, SX_test, Sy_test, Sall_labels

# Rename the best performing model to be generic, and get rid of the rest
def rename_best (path: str, best: int, models: int):
    """Rename the best model to remove index, and delete other models"""

    os.rename(f"{path}_{best}_metrics.txt", f"{path}_metrics.txt")
    os.rename(f"{path}_{best}_nn.pt", f"{path}_nn.pt")
    os.rename(f"{path}_{best}_plt.png", f"{path}_plt.png")

    # Remove other models, skipping best
    for i in range(models):
        if i == best:
            continue

        os.remove(f"{path}_{i}_metrics.txt")
        os.remove(f"{path}_{i}_nn.pt")
        os.remove(f"{path}_{i}_plt.png")

# Classify the samples in this file and add the classification to the output file
def classify_data (model: nn.Sequential, path: str, out: str, all_labels: list[str], 
                   features: list[str], drop: list[str] = [], labeled: bool = True, debug: bool = False):
    """Read the file from path and use the model to classify it. Copy the data and the 
    classificaitons to the output file"""

    # Load unlabeled data for the columns used by model
    data, count_cols = load_unlabeled(path, keep=features, debug=debug)
    print(f"Count cols: {count_cols}")

    # Get predictions
    model.eval()
    with torch.inference_mode():
        y_predictions = model(data)

    # Get predictions for each sample, and probabilties for each class
    predictions = prep_data(y_predictions)
    predictions = list(map(lambda x:all_labels[x], predictions))

    probabilities = y_predictions.softmax(dim=1).cpu().numpy()

    # Open the file and write back classificaitons and probabilities
    try:
        file = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Couldn't load {path}")
        exit(1)

    file["subCST"] = predictions

    for i, lbl in enumerate(all_labels):
        file[f"Pct {lbl}"] = probabilities[:,i]

    file.to_csv(f"{out}.csv", index=False)
    print(f"Output: {out}.csv")

# Return all the columns requested from data in order
# TODO
def get_cols(data, columns, data_features, strict = False):
    "Unimpemented; Unused"

    # For loose equality, standardize strings
    if not strict:
        columns = str_norm(columns)
        data_features = str_norm(data_features)

    #print(list(columns), list(data_features))
    for col in columns:
        print(col, col in data_features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains or uses a neural network classifier on microbiome count data")

    # Arguments for tool
    arguments = parser.add_argument_group("Arguments")
    arguments.add_argument("-itr", "--input-train", help="Path to train input data as csv", default=None)
    arguments.add_argument("-ite", "--input-test", help="Path to test input data as csv", default=None)
    arguments.add_argument("-p", "--path", help="Path to save/load neural network", default="classifier")
    arguments.add_argument("-cl","--classify", action=argparse.BooleanOptionalAction, help="Classify data provided by input-test", default=False)
    arguments.add_argument("-out", "--output", help="Path to output data as csv")
    arguments.add_argument("-tlr","--threshhold-lr", type=float, help="Train until lr is dropped to this level", default=0.00001)
    arguments.add_argument("-lr","--learning-rate", type=float, help="Learning rate for ai", default=0.01)
    arguments.add_argument("-me","--max-epochs", type=int, help="Maximum number of epochs to train for. 50000 is default", default=50000)
    arguments.add_argument("-t","--train", action=argparse.BooleanOptionalAction, help="Should a model be trained based on input data?", default=True)
    arguments.add_argument("-c","--continue-train", action=argparse.BooleanOptionalAction, help="Should a saved model be trained more?", default=False)
    arguments.add_argument("-m","--metrics-interval", type=int, help="How many epochs should training metrics be taken?", default=50)
    arguments.add_argument("-l","--loss", help="Loss function. ce (default), nll, or kld", default="ce")
    arguments.add_argument("-lo","--load", action=argparse.BooleanOptionalAction, help="Load model to train a simpler one", default=False)
    arguments.add_argument("-o","--optim", help="Optimizer. sgd, or adam (default)", default="adam")
    arguments.add_argument("-li","--linear", action=argparse.BooleanOptionalAction, help="Don't use ReLU?", default=False)
    arguments.add_argument("-pa","--patience", type=int, help="How many stagnant epochs to wait to cut lr?", default=100)
    arguments.add_argument("-sd","--seed", type=int, help="Seed rng", default=None)
    arguments.add_argument("-hn","--hidden-neurons", type=int, help="Number of hidden layers to use. Default is (2/3)*in_featres + classes", default=None)
    arguments.add_argument("-dbg","--debug", action=argparse.BooleanOptionalAction, help="Show verbose debugging and graphs", default=True)
    arguments.add_argument("-ts","--train-simple", action=argparse.BooleanOptionalAction, help="Train a version based on signigicant parameters", default=False)
    arguments.add_argument("-ta","--test-accuracy", action=argparse.BooleanOptionalAction, help="Classify data provided by input-test and check accuracy", default=False)
    arguments.add_argument("-tm","--train-multiple", type=int, help="How many models should be trained? Picks best", default=1)
    arguments.add_argument("-i","--info", action=argparse.BooleanOptionalAction, help="Print info about a model", default=False)
    arguments.add_argument("-fc", "--focus-columns", help="Columns to be used for simple models, comma seperated")
    arguments.add_argument("-lb", "--labeled", action=argparse.BooleanOptionalAction, help="Is data for classification labeled", default=False)
    arguments.add_argument("-n", "--normalizing-function", help="Method to use for normalizing data. none (default), log, tmm, rle", default="none")
    arguments.add_argument("-rr", "--regex-remove", help="Method to use for normalizing data. none (default), log, tmm, rle", default="")


    # Parse arguments
    args = parser.parse_args()

    # Set args to corresponding variables and preprosses if necessary
    linear = args.linear
    train_model = args.train
    continue_train = args.continue_train
    debug = args.debug
    path = args.path
    lr = args.learning_rate
    max_epochs = args.max_epochs
    metrics_interval = args.metrics_interval
    thresh = args.threshhold_lr
    train_simple = args.train_simple    
    classify = args.classify
    test_accuracy = args.test_accuracy
    info = args.info
    labeled = args.labeled
    norm_fn = args.normalizing_function
    load = args.load

    regex_remove = args.regex_remove.split(",")

    if args.seed is not None: torch.manual_seed(args.seed)

    if args.focus_columns != None: simple_cols = args.focus_columns.split(",")
    else: simple_cols = None

    # Override default arg
    if test_accuracy: train_model = False
    if classify: train_model = False
    if info: train_model = False
    
    if train_model:
        if args.input_train == None:
            print("ERR: Input Train required for training")
            exit(1)

        # Load data from supplied path
        X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, keys = load_data(args.input_train, args.input_test,
                                                                                       debug=debug, norm=norm_fn,
                                                                                       regex_remove=regex_remove)

        # Determine hidden neurons as (2/3)*input + output
        if args.hidden_neurons == None:
            hidden = int(round(len(X_train[0]) * (2/3) + len(y_train[0])))
            if debug: print(f"DBG: Using {hidden} hidden layers")
        else:
            hidden = args.hidden_neurons

        # Set up model
        if continue_train:
            # Improve existing model. Load it and train
            if debug: print(f"Continuing to train {path + '_nn.pt'}")
            classifier, structure, optim = load_model(path, debug=debug)
            train(classifier, X_train, y_train, X_test, y_test, lr, max_epochs, metrics_interval, thresh, 
                  args.loss, args.optim, linear, all_labels, ordered_prevelence, path, structure, keys, 
                  optim=optim, patience=args.patience, debug=debug)

        elif load:
            if debug: print(f"Loading {path + '_nn.pt'}")
            # Fall though, code immediately after this loads modelk from path regaurdless of choice
         
        else:
            # Train a model from scratch
            if debug: print("Training fresh model")
            accuracies = []
            for i in range(args.train_multiple):
                classifier, structure, optim = generate_model(linear, len(X_train[0]), hidden, len(y_train[0]), debug)
                acc = train(classifier, X_train, y_train, X_test, y_test, lr, max_epochs, metrics_interval, thresh, 
                            args.loss, args.optim, linear, all_labels, ordered_prevelence, f"{path}_{i}", structure, keys, 
                            optim=optim, patience=args.patience, debug=debug)
                
                accuracies.append(acc)

            best_model = accuracies.index(max(accuracies))
            rename_best(path, best_model, args.train_multiple)
            
        # Latest might not be best performer (which is what gets saved)
        classifier, _, _ = load_model(path, debug=debug)
        

        # Evaluate the model and plot the correlations
        if debug:
            test(classifier, X_test, y_test, all_labels)
            plot_correlations(classifier, X_test, y_test, all_labels, keys)

        if train_simple:
            if debug: print("Training simplified")
            # Find the sorted imporrtances of each feature, then train and test a network on the importnant features
            sorted_importances = feature_importance(classifier, X_test, y_test, keys)

            Simple_classifier, SX_test, Sy_test, Sall_lbls = train_simpler_model(args.input_train, args.input_test, 
                                                                                 sorted_importances, 1, lr, max_epochs, 
                                                                                 metrics_interval, thresh, args.loss, 
                                                                                 args.optim, linear, path, simple_cols,
                                                                                 patience=args.patience, debug=debug,
                                                                                 models = args.train_multiple, norm=norm_fn,
                                                                                 regex_remove=regex_remove)

            # Evaluate model and plot correlations
            if debug:
                test(Simple_classifier, SX_test, Sy_test, Sall_lbls)
                plot_correlations(Simple_classifier, SX_test, Sy_test, Sall_lbls, list(sorted_importances.keys()))

    elif classify:
        # TODO
        #data, data_features = load_unlabeled(args.input_test)
        #classifier, _, _, features = load_model(path, return_features=True)
        #
        #d = {'g_streptococcus':"streptococcus_gallolyticus", "g_fastidiosipila":"", "g_bifidobacterium":""}
        #
        #data = get_cols(data, list(features), list(data_features))
        # TODO: Add model predictions to the file
        #print(list(features), list(data_features))

        classifier, _, _, features, all_labels = load_model(path, return_features = True, debug=debug)
        print(f"Lbls:{all_labels}")
        classify_data(classifier, args.input_test, args.output, all_labels, features, labeled=labeled, debug=debug)

    elif test_accuracy:
        # Load model and data from supplied path
        if debug: print("Loading model")
        classifier, _, _, features, _ = load_model(path, return_features = True, debug=debug)

        #print(features)
        X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, keys = \
                        load_data(args.input_train, args.input_test, keep=features, debug=debug, regex_remove=regex_remove, norm=norm_fn)
 
        # Test model and evaluate it
        test(classifier, X_test, y_test, all_labels)
        plot_correlations(classifier, X_test, y_test, all_labels, keys)

        print("Correct guesses\n1st guess, 2nd guess, ...")
        print(top_n_accuracy(classifier, X_train, y_train))

    elif info:
        # Load model and print info
        model_state, structure, all_labels, features, optim_type, lr, optim_state = get_model_info(path)

        # Make len and .join() work for invalid models
        if features is None: features = []
        if all_labels is None: all_labels = []

        # Print info for user
        print(f"Model at [{path}]'s structure is {structure}")
        print(f"It requires {len(features)} features: {', '.join(features)}") 
        print(f"It predicts {len(all_labels)} classes: {', '.join(all_labels)}") 
        print(f"Optimizer used: {optim_type} Final learning rate: {lr}") 