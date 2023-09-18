# Alexander Symons | Aug 25 2023 | nn_classifier.py
# This file trains and deploys a neural network classifier for microbiome data based on 
# count data for each bacteria
# Inputs: A path to a csv file containing the count data
# Outputs: A file containing parameters for the neural network and predictions for each 
#          sample in the input data

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

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
    skl = True
except:
    print("Optional package sklearn not available")
    skl = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.widgets import Button
    mpl = True
except:
    print("Optional package matplotlib not available")
    mpl = False

# Set up GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")

losses = {"nll":nn.NLLLoss, "ce":nn.CrossEntropyLoss, "kld":nn.KLDivLoss}
optims = {"sgd": torch.optim.SGD, "adam":torch.optim.Adam}

# Load data from paths to csv test and training data files.
def load_data(train_path, test_path):
    """Loads data from supplied (str) paths to csv training and testing data. Returns
    normalized 'x' input data as a tensor on the device, 'y' output data as one hot tensors
    corresponding to that classes index in all_labels, a sorted list encoding the order of the classes,
    ordered_prevelence which is a tensor on the device with adjused prevelences of each class, and 
    the column names of those containing input data."""

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

    # Create helper functions for formatting CST 'labels' for the neural network
    if set(dftr["HC_subCST"]) != set(dfte["HC_subCST"]):
        print("Training and test set do not contain same CSTs")
        exit()

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

    # TODO: Test this with more metrics. It will affect how the model learns rarer classes <---- Look more into IV-C4
    # TODO: Try different measures of prevelance. Giving rarer classes such a bigger importance will
    # impact the accuracy of more common classes a lot
    # Get in order of what index each label is in training data
    ordered_prevelence = torch.tensor([1/entries[all_labels[i]] for i in range(len(all_labels))]).to(device)
    ordered_prevelence *= 1/ordered_prevelence.min()

    return X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, count_columns

# Create a Neural Net 
# Current layers do sqrt(in*out) and split in half, and do it again. 51=sqrt(101*13) 101=sqrt(199*51) 26=sqrt(51*13)
def generate_model(linear, train_features, hidden_features, classes, old=False):
    """Generates a fresh model based on model linearity, number of training features, number of 
    hidden features, and number of output features. Returns the model, a string representation
    of its structure, and None (for optimizer)"""

    if old:
        # Old architecture is WAY overcomplicated
        if not linear:
            structure = f"OLD Non-linear {train_features} -> {classes}"
            print(structure)
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
            print(structure)
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
            print(structure)
            classifier = nn.Sequential(
                nn.Linear(in_features=train_features, out_features=hidden_features),
                nn.ReLU(),
                #nn.Linear(in_features=hidden_features, out_features=hidden_features),
                #nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=classes),
                nn.Softmax(dim=1)
            ).to(device)
        else:
            structure = f"NEW Linear {train_features} -> {hidden_features} -> {classes}"
            print(structure)
            classifier = nn.Sequential(
                nn.Linear(in_features=train_features, out_features=hidden_features),
                #nn.Linear(in_features=hidden_features, out_features=hidden_features),
                nn.Linear(in_features=hidden_features, out_features=classes),
                nn.Softmax(dim=1)
            ).to(device)

    # Model, structure:str, optimizer (None)
    return classifier, structure, None

# Define function to find accuracy of model
def accuracy_test(lbls, predictions):
    # Do argmax to get index, so as not to do torch.eq on a 2d array
    correct = torch.eq(torch.Tensor([lbl.argmax() for lbl in lbls]), torch.Tensor([p.argmax() for p in predictions])).sum().item()
    return (correct / len(predictions)) * 100

# Evaluate feature importance using the model
def feature_importance(model, data, lbls, features):
    with torch.inference_mode():
        test_pred = classifier(data)
    
    standard_score = accuracy_test(lbls, test_pred)
    
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
            test_pred = classifier(data_cpy)

        permuted_score = accuracy_test(lbls, test_pred)

        # Find difference between scores and save it
        importances[feat] = standard_score - permuted_score

    sorted_importances = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)
    print(sorted_importances)

def train(classifier, X_train, y_train, X_test, y_test, lr, max_epochs, metrics_interval, 
          thresh, loss_type, optim_type, linear, all_labels, ordered_prevalence, path, structure, optim=None, debug=False):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=50, verbose=True)

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

                print(f"Epoch: {epoch} ({((epoch/max_epochs)*100):.2f}%), Loss: {loss}, ", end='')
                print(f"Test: {test_loss}, Acc: {test_accuracy}, lr:{optim.state_dict()['param_groups'][0]['lr']}")

                # Save if new best
                if test_accuracy > max_acc:
                    max_acc = test_accuracy
                    torch.save({"model": classifier.state_dict(),
                                "structue": structure,
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
            if optim.state_dict()['param_groups'][0]['lr'] <= thresh:
                break

    # Save model
    #torch.save(obj=classifier.state_dict(), f=path+"_nn.pt")

    # Calculate time
    time_now = time.time()
    time_taken = f"{int((time_now - start_time) / 3600)}:{int(((time_now - start_time) / 60) % 60)}:{int((time_now - start_time) % 60)}"

    # Write metrics
    with torch.inference_mode():

        with open(path + "_metrics.txt", "w") as f:
            print(f"Final Max Accuracy: {max_acc}% in {time_taken}")
            f.write(f"Final Max Accuracy: {max_acc}% in {time_taken}\n")

            print(f"Test Config: lr: {lr}, linear: {linear}, loss_fn: {loss_type}, optim: {optim_type}\n")
            f.write(f"Test Config: lr: {lr}, linear: {linear}, loss_fn: {loss_type}, optim: {optim_type}\n")

            for i in range(len(epoch_count)):
                f.write(f"Epoch: {epoch_count[i]}, Train loss: {loss_values[i]}, Test Loss: {test_losses[i]}, Accuracy: {test_accuracies[i]}\n")

            f.write("Cases: \n")
            for i in range(10):
                f.write(f"Actual: {i_to_lbl(y_test[i])}, Predicted: {i_to_lbl(y_predictions[i])}\n")

        if mpl:
            prep = lambda x:list(map(lambda y:y.cpu().item(), x))

            plt.plot(epoch_count, prep(loss_values))
            plt.plot(epoch_count, prep(test_losses))
            #plt.plot(epoch_count, test_accuracies)
            plt.legend(["Train Loss", "Test Loss", "Test Accuracy"])
            plt.savefig(f"{path}_plt")

    return max_acc
    
# Pepare data. Convert from one-hot cuda tensor to scalar np array
def prep_data(data):
    prepped = data.cpu()
    prepped = prepped.argmax(dim=1)
    return prepped.numpy()

# Test model, taking various metrics
def test(model, X_test, y_test, all_labels):
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
    print(f"accuracy: {accuracy_test(y_test, y_predictions):.2f}%")

    print('  '.join(all_labels))

    conf_mat = confusion_matrix(lbls, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=all_labels)
    disp.plot()
    plt.show()
    print(conf_mat)

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

    # Draw fresh scatterplot with the current features
    def update_scatter(self):
        self.ax.clear()

        s = self.ax.scatter(self.X_test[:,self.feature_1], self.X_test[:,self.feature_2], c=self.y_test, cmap=plt.cm.plasma)

        self.fig.legend(s.legend_elements()[0], self.labels,
                        loc="center right", title="CST")
        
        self.ax.set_xlabel(self.keys[self.feature_1])
        self.ax.set_ylabel(self.keys[self.feature_2])
        self.ax.set_title(f"Features: {self.keys[self.feature_1]} x {self.keys[self.feature_2]}")
        
        self.fig.canvas.draw()

    # Select feature1/2 to plot and compare
    def next_f1(self, event):
        self.feature_1 += 1
        self.update_scatter()

    def prev_f1(self, event):
        self.feature_1 -= 1
        self.update_scatter()

    def next_f2(self, event):
        self.feature_2 += 1
        self.update_scatter()

    def prev_f2(self, event):
        self.feature_2 -= 1
        self.update_scatter()

# Plot boundary line of two features
def plot_correlations(model, X_test, y_test, all_labels, keys):
    if not plt:
        print("Matplot required for plot_correlations.")
        return

    # Get data to explore
    model.eval()
    with torch.inference_mode():
        predictions = model(X_test)
        predictions = prep_data(predictions)

    classes = len(y_test[0])

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

    button_next_f1 = Button(next_f1_ax, "Next feature1", color='green', hovercolor='blue')
    button_prev_f1 = Button(prev_f1_ax, "Previous featue1", color='green', hovercolor='blue')
    button_next_f2 = Button(next_f2_ax, "Next feature2", color='green', hovercolor='blue')
    button_prev_f2 = Button(prev_f2_ax, "Previous feature2", color='green', hovercolor='blue')
    
    # Register click event callbacks to the plotter's handler functions
    button_next_f1.on_clicked(plotter.next_f1)
    button_prev_f1.on_clicked(plotter.prev_f1)
    button_next_f2.on_clicked(plotter.next_f2)
    button_prev_f2.on_clicked(plotter.prev_f2)    

    plt.show()


# Load model at path from the path_nn.pt
def load_model(path):
    # Load 'checkpoint'
    try:
        checkpoint = torch.load(path + "_nn.pt")
    except:
        print(f"Couldn't read {path}_nn.pt")
        exit()

    # Validate architecture type, linearity, and layer sizes
    structure = checkpoint["structue"].split(" ")
    
    old_arch = True if structure[0] == "OLD" else False if structure[0] == "NEW" else None
    linear = True if structure[1] == "Linear" else False if structure[1] == "Non-linear" else None

    if old_arch == None or linear == None:
        print(f"Erroneous structure data in {path}_nn.pt; Old: {old_arch}, Linear: {linear}")
        exit()

    try:
        input_size = int(structure[2])
        hidden_size = int(structure[4])
        output_size = int(structure[6])
    except TypeError:
        print(f"Erroneous structure data in {path}_nn.pt; layers must be int -> int -> int")

    # Create the classifier and load its state from nn.pt
    classifier, structure, _ = generate_model(linear, input_size, hidden_size, output_size)
    classifier.load_state_dict(checkpoint["model"])

    # Load optimizer
    optim = optims[checkpoint["optim_type"]](params=classifier.parameters(), lr=checkpoint["lr"])
    optim.load_state_dict(checkpoint["optim"])

    # Model, structure (str), optimizer
    return classifier, structure, optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains or uses a neural network classifier on microbiome count data")

    # Arguments for tool
    arguments = parser.add_argument_group("Arguments")
    arguments.add_argument("-itr", "--input-train", help="Path to train input data as csv", required=True)
    arguments.add_argument("-ite", "--input-test", help="Path to test input data as csv", required=True)
    arguments.add_argument("-p", "--path", help="Path to save/load neural network", default="classifier")
    arguments.add_argument("-tlr","--threshhold-lr", help="Train until lr is dropped to this level", default=0.00001)
    arguments.add_argument("-lr","--learning-rate", help="Learning rate for ai", default=0.001)
    arguments.add_argument("-me","--max-epochs", help="Maximum number of epochs to train for. None is default", default=None)
    arguments.add_argument("-t","--train", help="Should a model be trained based on input data?", default="True")
    arguments.add_argument("-c","--continue-train", help="Should a saved model be trained more?", default="False")
    arguments.add_argument("-m","--metrics-interval", help="How many epochs should training metrics be taken?", default=50)
    arguments.add_argument("-l","--loss", help="Loss function. ce (default), nll, or kld", default="ce")
    arguments.add_argument("-o","--optim", help="Optimizer. sgd (default), or adam", default="adam")
    arguments.add_argument("-li","--linear", help="Don't use ReLU?", default="False")
    arguments.add_argument("-sd","--seed", help="Seed rng", default=None)
    arguments.add_argument("-hl","--hidden-layers", help="Number of hidden layers to use. Default is (2/3)*in_featres + classes", default=None)
    arguments.add_argument("-dbg","--debug", help="Show verbose debugging and graphs", default="N")


    # Parse arguments
    args = parser.parse_args()

    # Parse all boolean arguments using string comparison
    linear = args.linear == "True"
    train_model = args.train == "True"
    continue_train = args.continue_train == "True"

    debug = True if args.debug == "N" else False if args.debug == "Y" else None

    path = args.path

    # Ensure all arguments are correct type
    if args.seed is not None:
        try:
            seed = int(args.seed)
        except TypeError:
            print("Seed must be an int")
            exit()

        torch.manual_seed(seed)

    if train_model:
        try:
            lr = float(args.learning_rate)
        except TypeError:
            print("Learning rate must be float")
            exit()

        try: 
            max_epochs = int(args.max_epochs)
        except TypeError:
            print("max epochs must be int")
            exit()

        try: 
            metrics_interval = int(args.metrics_interval)
        except TypeError:
            print("metrics interval must be int")
            exit()

        try: 
            thresh = float(args.threshhold_lr)
        except TypeError:
            print("accuracy must be float")
            exit()

    # Load data from supplied path
    X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, keys = load_data(args.input_train, args.input_test)

    # Determine hidden layers
    if args.hidden_layers == None:
        hidden = int(round(len(X_train[0]) * (2/3) + len(y_train[0])))
        print(hidden)
    else:
        try: 
            hidden = int(args.hidden_layers)
        except TypeError:
            print("Hidden layers must be int")
            exit()

    # Set up model
    if train_model:
        if continue_train:
            # Improve existing model
            print(f"Continuing to train {path + '_nn.pt'}")
            classifier, structure, optim = load_model(path)

        else:
            # Train a model from scratch
            print("Training fresh model")
            classifier, structure, optim = generate_model(linear, len(X_train[0]), hidden, len(y_train[0]))
        
        train(classifier, X_train, y_train, X_test, y_test, lr, max_epochs, metrics_interval, thresh, 
              args.loss, args.optim, linear, all_labels, ordered_prevelence, path, structure, optim=optim)
        
        # Latest might not be best performer (which is what gets saved)
        classifier, _, _ = load_model(path)
        
    else:
        print("Loading model")
        classifier, _, _ = load_model(path)
    
    # Evaluate the model and plot the correlations
    #test(classifier, X_test, y_test, all_labels)
    #plot_correlations(classifier, X_test, y_test, all_labels, keys)

    feature_importance(classifier, X_test, y_test, keys)