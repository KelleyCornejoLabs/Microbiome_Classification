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

    # TODO: Normalize from calculated totals not total row

    # Load and validate the input data
    try:
        dftr = pd.read_csv(train_path)
    except FileNotFoundError:
        print("Could not load input training data")
        exit()

    try:
        dfte = pd.read_csv(test_path)
    except FileNotFoundError:
        print("Could not load input test data")
        exit()

    def check_keys(df_keys, in_type):
        if False in [(key in df_keys) for key in ["sampleID", "read_count", "HC_subCST"]]:
            print(f"{in_type} input does not contain sampleID, read_count, or HC_subCST")
            exit()

    check_keys(list(dftr.keys()), "Training")
    check_keys(list(dfte.keys()), "Testing")

    # Drop columns for simplified model
    ignore = False
    if drop != []:
        for col in drop:
            try:
                dftr = dftr.drop(columns=[col])
            except KeyError:
                print(f"Failed to drop {col} from training set")
                if not ignore:
                    ignore = bool(input("Continue anyways?"))
                    if ignore: continue
                    exit()

            try:
                dfte = dfte.drop(columns=[col])
            except KeyError:
                print(f"Failed to drop {col} from test set")
                if not ignore:
                    ignore = bool(input("Continue anyways?"))
                    if ignore: continue
                    exit()

    # Create helper functions for formatting CST 'labels' for the neural network
    if set(dftr["HC_subCST"]) != set(dfte["HC_subCST"]):
        print("Training and test set do not contain same CSTs")
        exit()

    # Sort to standardize data loading
    all_labels = sorted(list(set(dftr["HC_subCST"])))

    # Encode and decode labels as one-hot vector corresponding to their index in all_labels
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
        exit()


    if keep != None:
        for col in count_columns:
            if not col in keep:
                normalized_test_data = normalized_test_data.drop(columns=[col])
                normalized_train_data = normalized_train_data.drop(columns=[col])


    # Regex to remove g_*. Use "^[a-z]_"
    if regex_remove != ['']:
        for regex in regex_remove:
            print(f"Removing from regex {regex}: {', '.join(list(normalized_train_data.filter(regex=regex)))}")
            normalized_train_data = normalized_train_data[normalized_train_data.columns.drop(list(normalized_train_data.filter(regex=regex)))]
            normalized_test_data = normalized_test_data[normalized_test_data.columns.drop(list(normalized_test_data.filter(regex=regex)))]

    # TODO: Handle if two files have same columns in different order?

    # Get columns to normalize, remove if not to be kept
    count_columns = normalized_train_data.columns

    # Normalize the columns based on count data
    # Adjust for sample "quality" always
    for column in count_columns:
        normalized_train_data[column] /= dftr["read_count"]
        normalized_test_data[column] /= dfte["read_count"]

    if norm == "log":
        for column in count_columns:
            normalized_train_data[column] /= list(map(lambda x:math.log10(x+0.001), normalized_train_data[column]))
            normalized_test_data[column] /= list(map(lambda x:math.log10(x+0.001), normalized_test_data[column]))
    elif norm == "tmm":
        if not cnrm:
            print("conorm package required for TMM normalization")
            exit(1)

        normalized_train_data = conorm.tmm(normalized_train_data.T).T
        normalized_test_data = conorm.tmm(normalized_test_data.T).T

    # Format for neural net
    training_data = torch.tensor(normalized_train_data[count_columns].to_numpy()).type(torch.float).to(device)
    testing_data = torch.tensor(normalized_test_data[count_columns].to_numpy()).type(torch.float).to(device)

    

    # Split training data into the train and test sets  
    if debug: print(f"Training split: {len(training_data)}, Testing split: {len(testing_data)}")

    X_train, y_train = training_data, train_labels
    X_test, y_test = testing_data, test_labels

    # if split:
    #     print("Training on all given data. No data will be reserved for metrics")
    #     X_test = X_train
    #     y_test = y_train

    if debug: print(f"Sizes: train: {len(X_train), len(y_train)}, test: {len(X_test), len(y_test)}")

    if debug: print(f"First 5 labels of train: {', '.join(map(i_to_lbl, y_train[:5]))}")

    # Found out what proportion of the data each CST makes
    entries = dftr.groupby(["HC_subCST"]).count()["sampleID"]

    # TODO: Test this with more metrics. It will affect how the model learns rarer classes <---- Look more into IV-C4
    # TODO: Try different measures of prevelance. Giving rarer classes such a bigger importance will
    # impact the accuracy of more common classes a lot
    # Get in order of what index each label is in training data
    ordered_prevelence = torch.tensor([1/entries[all_labels[i]] for i in range(len(all_labels))]).to(device)
    ordered_prevelence *= 1/ordered_prevelence.min()

    return X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, count_columns

# Load unlabeled data for classification by model
def load_unlabeled(path: str, drop: list[str]|None = [], keep: list[str]|None = None, 
                   debug:bool = False) -> tuple[torch.Tensor, list[str]]:
    """Load data without labels to be fed to model for classification. A list of column names
    to explicity keep (drop everything else) or provide a list of column names to drop"""

    # Read file
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Couldn't find file at path '{path}'")
        exit(1)

    # Only keep columns requested 
    if keep != None:
        # Drop all non-requested columns
        for col in list(df.columns):
            if not col in keep:
                if debug: print(f"Dropping {col}")
                drop += [col]
            else:
                if debug: print("KEEP", col)

    # Drop columns if required=
    if drop != []:
        loaded = df
        for col in drop:
            try:
                loaded = loaded.drop(columns=[col])
            except ValueError:
                print(f"Failed to drop {col} from training set")
                exit()
    else:
        # Don't feed ID and read counts to net. Removed previously if keep
        loaded = df.drop(columns=["sampleID", "read_count"])

    # Normalize
    count_columns = loaded.columns
    for column in count_columns:
        loaded[column] /= df["read_count"]

    # Return as float tensor on device
    return torch.tensor(loaded[count_columns].to_numpy()).type(torch.float).to(device), count_columns
    

# Create a Neural Net 
# Current layers do sqrt(in*out) and split in half, and do it again. 51=sqrt(101*13) 101=sqrt(199*51) 26=sqrt(51*13)
def generate_model(linear: bool, train_features: int, hidden_features: int, classes: int, dbg: bool, 
                   old:bool = False) -> tuple[nn.Sequential, str, None]:
    """Generates a fresh model based on model linearity, number of training features, number of 
    hidden features, and number of output features. Returns the model, a string representation
    of its structure, and None (for optimizer)"""

    if old:
        # Old architecture is WAY overcomplicated, not reccomended
        if not linear:
            structure = f"OLD Non-linear {train_features} -> {classes}"
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
            structure = f"OLD Linear {train_features} -> {classes}"
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
                nn.ReLU(),
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=patience, verbose=debug)

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

# Store all data relating to data visualization and associated callbacks
class Plotter:
    feature_1 = 0
    feature_2 = 0

    def __init__(self, ax, fig, X_test, y_test, colors, keys, labels):
        self.ax = ax
        self.fig = fig
        self.X_test = X_test
        self.y_test = y_test
        self.colors = colors
        self.keys = keys
        self.labels = labels
        self.first_time = True

    # Draw fresh scatterplot with the current features
    def update_scatter(self):

        self.ax.clear()

        s = self.ax.scatter(self.X_test[:,self.feature_1], self.X_test[:,self.feature_2], c=self.y_test, cmap=plt.cm.plasma)
        
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

    plotter = Plotter(ax, fig, X_test, y_test, plt.cm.plasma, keys, all_labels) # Track plotting variables
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
def load_model(path: str, keys: None|list[str] = None, return_features: bool = False) -> \
    tuple[nn.Sequential, str, torch.optim.Optimizer, list[str]|None, list[str]|None]:
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
    classifier, structure, _ = generate_model(linear, input_size, hidden_size, output_size, debug)
    classifier.load_state_dict(checkpoint["model"])

    # Load optimizer
    optim = optims[checkpoint["optim_type"]](params=classifier.parameters(), lr=checkpoint["lr"])
    optim.load_state_dict(checkpoint["optim"])

    # Model, structure (str), optimizer, features (optional)
    if return_features: return classifier, structure, optim, list(checkpoint["features"]), list(checkpoint["all_labels"])
    else: return classifier, structure, optim

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
        unimportant_cols = [key for key, value in sorted_importances.items() if value < imporatance_threshold]
        
        # Determine new data containing the Significant columns
        SX_train, Sy_train, SX_test, Sy_test, Sall_labels, Sordered_prevelence, Skeys = load_data(train_path, test_path, 
                                                                                                  drop=unimportant_cols,
                                                                                                  debug=debug, norm=norm,
                                                                                                  regex_remove=regex_remove)
    else: # Or manually keep columns
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
    Sclassifier, _, _ = load_model(f"{path}_simplified")
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
                   features: list[str], drop: list[str] = [], labeled: bool = True):
    """Read the file from path and use the model to classify it. Copy the data and the 
    classificaitons to the output file"""

    # Load unlabeled data for the columns used by model
    data, count_cols = load_unlabeled(path, keep=features)

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
    standardize = lambda x:list(map(lambda y:y.lower().replace(' ','_'), x))
    if not strict:
        columns = standardize(columns)
        data_features = standardize(data_features)

    #print(list(columns), list(data_features))
    for col in columns:
        print(col, col in data_features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains or uses a neural network classifier on microbiome count data")

    # Arguments for tool
    arguments = parser.add_argument_group("Arguments")
    arguments.add_argument("-itr", "--input-train", help="Path to train input data as csv", default=None)
    arguments.add_argument("-ite", "--input-test", help="Path to test input data as csv", required=True)
    arguments.add_argument("-out", "--output", help="Path to output data as csv")
    arguments.add_argument("-p", "--path", help="Path to save/load neural network", default="classifier")
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
    arguments.add_argument("-hl","--hidden-layers", type=int, help="Number of hidden layers to use. Default is (2/3)*in_featres + classes", default=None)
    arguments.add_argument("-dbg","--debug", action=argparse.BooleanOptionalAction, help="Show verbose debugging and graphs", default=True)
    arguments.add_argument("-ts","--train-simple", action=argparse.BooleanOptionalAction, help="Train a version based on signigicant parameters", default=False)
    arguments.add_argument("-cl","--classify", action=argparse.BooleanOptionalAction, help="Classify data provided by input-test", default=False)
    arguments.add_argument("-ta","--test-accuracy", action=argparse.BooleanOptionalAction, help="Classify data provided by input-test and check accuracy", default=False)
    arguments.add_argument("-tm","--train-multiple", type=int, help="How many models should be trained? Picks best", default=1)
    arguments.add_argument("-i","--info", action=argparse.BooleanOptionalAction, help="Print info about a model", default=False)
    arguments.add_argument("-fc", "--focus-columns", help="Columns to be ignored for simple models, comma seperated")
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
            print("Input Train required for training")
            exit(1)

        # Load data from supplied path
        X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, keys = load_data(args.input_train, 
                                                                                           args.input_test, 
                                                                                           debug=debug, norm=norm_fn,
                                                                                           regex_remove=regex_remove)

        # Determine hidden layers as (2/3)*input + output
        if args.hidden_layers == None:
            hidden = int(round(len(X_train[0]) * (2/3) + len(y_train[0])))
            if debug: print(f"Using {hidden} hidden layers")
        else:
            hidden = args.hidden_layers

        # Set up model
        if continue_train:
            # Improve existing model
            if debug: print(f"Continuing to train {path + '_nn.pt'}")
            classifier, structure, optim = load_model(path)
            train(classifier, X_train, y_train, X_test, y_test, lr, max_epochs, metrics_interval, thresh, 
                  args.loss, args.optim, linear, all_labels, ordered_prevelence, path, structure, keys, 
                  optim=optim, patience=args.patience, debug=debug)

        elif load:
            if debug: print(f"Loading {path + '_nn.pt'}")
         
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
        classifier, _, _ = load_model(path)
        

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

        classifier, _, _, features, all_labels = load_model(path, return_features = True)
        classify_data(classifier, args.input_test, args.output, all_labels, features, labeled=labeled)

    elif test_accuracy:
        # Load model and data from supplied path
        if debug: print("Loading model")
        classifier, _, _, features, _ = load_model(path, return_features = True)

        #print(features)
        X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, keys = \
                        load_data(args.input_train, args.input_test, keep=features, debug=debug, regex_remove=regex_remove)
 
        # Test model and evaluate it
        test(classifier, X_test, y_test, all_labels)
        plot_correlations(classifier, X_test, y_test, all_labels, keys)

        print("Correct guesses\n1st guess, 2nd guess, ...")
        print(top_n_accuracy(classifier, X_train, y_train))

    elif info:
        # Load model and print info
        _, structure, _, features, all_labels = load_model(path, return_features = True)
        print(f"Model at [{path}]'s structure is {structure}")
        print(f"It requires {len(features)} features: {', '.join(features)}") 
        print(f"It predicts {len(all_labels)} classes: {', '.join(all_labels)}") 