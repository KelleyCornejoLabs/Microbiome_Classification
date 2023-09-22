import sys

import nn_classifier

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
    import matplotlib.pyplot as plt
except:
    print("Required package matplotlib not available")
    exit()

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except:
    print("Required package sklearn not available")
    exit()

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

    # When converting to labels, just convert between string and its index in all_labels.
    def i_to_lbl(i):
        return i.argmax()

    def lbl_to_i(lbl):
        return all_labels.index(lbl)

    # Generate label data
    test_labels = np.array([lbl_to_i(x) for x in list(dfte["HC_subCST"])])
    train_labels = np.array([lbl_to_i(x) for x in list(dftr["HC_subCST"])])

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

    # Split training data into the train and test sets  
    print(f"Training split: {len(normalized_train_data)}, Testing split: {len(normalized_test_data)}")

    X_train, y_train = normalized_train_data, train_labels
    X_test, y_test = normalized_test_data, test_labels

    # if split:
    #     print("Training on all given data. No data will be reserved for metrics")
    #     X_test = X_train
    #     y_test = y_train

    print(f"Sizes: train: {len(X_train), len(y_train)}, test: {len(X_test), len(y_test)}")

    # Found out what proportion of the data each CST makes
    entries = dftr.groupby(['HC_subCST']).count()['sampleID']

    # TODO: Test this with more metrics. It will affect how the model learns rarer classes
    # Get in order of what index each label is in training data
    ordered_prevelence = [1/entries[all_labels[i]] for i in range(len(all_labels))]
    scalar = 1/min(ordered_prevelence).item()
    ordered_prevelence = list(map(lambda x:x*scalar, ordered_prevelence))

    return X_train, y_train, X_test, y_test, all_labels, ordered_prevelence

# Define function to find accuracy of model
def accuracy_test(lbls, predictions):
    # Do argmax to get index, so as not to do torch.eq on a 2d array
    correct = sum(np.equal(lbls, predictions))
    return (correct / len(predictions)) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates or uses random forest classifier on microbiome count data")

    # Arguments for tool
    arguments = parser.add_argument_group("Arguments")
    arguments.add_argument("-itr", "--input-train", help="Path to train input data as csv", required=True)
    arguments.add_argument("-ite", "--input-test", help="Path to test input data as csv", required=True)

    # Parse arguments
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, all_labels, _, _ = nn_classifier.load_data(args.input_train, args.input_test)

    X_train = X_train.cpu().numpy()
    X_test = X_test.cpu().numpy()

    y_train = y_train.argmax(dim=1).cpu().numpy()
    y_test = y_test.argmax(dim=1).cpu().numpy()

    # TODO: get clas weight working
    # Maybe have more estimators that look at a smaller set of features? To try and find several linear differences instead of looking so broadly?
    model = RandomForestClassifier(n_estimators=100, max_features="sqrt")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    incorrect = [':'.join([all_labels[i], all_labels[j]]) for i,j in zip(y_test, predictions) if i != j]

    print(len(incorrect), ', '.join(incorrect))

    print(f"Accuracy: {accuracy_test(y_test, predictions)}")

    conf_mat = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=all_labels)
    disp.plot()
    plt.show()
    print(f"Conf mat: \n{conf_mat}")