import sys
import argparse
import nn_classifier
import time
import math

# Arguments for tool
parser = argparse.ArgumentParser(description="Trains several neural classifiers to find best hyperparameters")

arguments = parser.add_argument_group("Arguments")
arguments.add_argument("-itr", "--input-train", help="Path to train input data as csv", required=True)
arguments.add_argument("-ite", "--input-test", help="Path to test input data as csv", required=True)
arguments.add_argument("-e", "--epochs", help="Path to test input data as csv", required=True)
arguments.add_argument("-t", "--tests", help="Path to test input data as csv", default=3)
arguments.add_argument("-o", "--optims", help="Comma seperated list of optimizers to test", required=True)
arguments.add_argument("-l", "--losses", help="Comma seperated lsit of losses to test", required=True)
arguments.add_argument("-lr", "--learning-rates", help="Comma seperated lsit of lrs to test", required=True)
arguments.add_argument("-nl", "--non-linear", help="Test non-linear", required=False)
arguments.add_argument("-p", "--path", help="Path to save findings", default="results")
arguments.add_argument("-pt", "--path-tests", help="Path to Tests", default="tests/results")
arguments.add_argument("-n", "--norms", help="Norm methods", default="none")
arguments.add_argument("-r", "--regexs", help="regexs", default="none")
arguments.add_argument("-d", "--droprates", help="regexs", default="none")
arguments.add_argument("-f", "--features", help="regexs", default="none")


args = parser.parse_args()

# Validate arguments
try:
    max_epochs = int(args.epochs)
except TypeError:
    print("epochs must be int")
    sys.exit(1)

try:
    tests = int(args.tests)
except TypeError:
    print("tests must be int")
    sys.exit(1)

optims = args.optims.split(",")
for optim in optims:
    if not optim in nn_classifier.optims.keys():
        print("optim must be from", ' '.join(nn_classifier.optims.keys()))
        sys.exit(1)

losses = args.losses.split(",")
for loss in losses:
    if not loss in nn_classifier.losses.keys():
        print("loss must be from", ' '.join(nn_classifier.losses.keys()))
        sys.exit(1)

norms = args.norms.split(",")
for n in norms:
    if not n in ("none", "log", "tmm"):
        print("norm bad")
        sys.exit(1)

regex = args.regexs.split(",")

drops = [float(x) for x in args.droprates.split(",")]

features = [int(x) for x in args.features.split(",")]

try:
    lrs = [float(lr) for lr in args.learning_rates.split(",")]
except TypeError:
    print("Each learning rate must be a float")
    sys.exit(1)

linear = not args.non_linear == "True"
performances = {}

for n in norms:
    # Get testing data
    X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, keys = nn_classifier.load_data(args.input_train, args.input_test, norm=n, regex_remove=regex)

    for optim in optims:
        for loss in losses:
            for lr in lrs:
                for dr in drops:
                    for f in features:
                        model_performances = []
                        for test in range(tests):
                            path = f"{args.path_tests}_{optim}_{loss}_{lr}_{n}_{dr}_{f}_{test}".replace(".",",") # EU style for UNIX machines
                            path += "_metrics.txt"
                            with open(path, "r") as file:
                                model_performances.append(float(file.readline().split(" ")[3][:-1]))

                        performances[f"{optim}_{loss}_{lr}_{n}_{dr}_{f}"] = sum(model_performances)/tests

print(performances)
print(f"Best performer: {sorted(performances.items(), key=lambda x:x[1], reverse=True)[0]}")