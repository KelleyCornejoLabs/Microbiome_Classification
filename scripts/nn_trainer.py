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

try:
    lrs = [float(lr) for lr in args.learning_rates.split(",")]
except TypeError:
    print("Each learning rate must be a float")
    sys.exit(1)

linear = not args.non_linear == "True"
print("Linear", linear)
print(f"Epochs {max_epochs}")

# TODO: Try pruning algorithms
start = time.time()

# Try each option
performaces = {}
for optim in optims:
    for loss in losses:
        for lr in lrs:
            for n in norms:
                
                # Get testing data
                X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, keys = nn_classifier.load_data(args.input_train, args.input_test, norm=n, regex_remove=regex)

                model_performances = []
                for test in range(tests):
                    path = f"{args.path_tests}_{optim}_{loss}_{lr}_{n}_{test}".replace(".",",") # EU style for UNIX machines
                    model, struct, _ = nn_classifier.generate_model(linear, len(X_train[0]), 146, len(y_train[0]), True)
                    acc = nn_classifier.train(model, X_train, y_train, X_test, y_test, lr, max_epochs, 1000, 
                                              0.000001, loss, optim, linear, all_labels, ordered_prevelence, path, struct, keys, debug=True, patience=10000)
                    model_performances.append(acc)

                print(f"Done: test {test}/{tests}, norm {norms.index(n)}/{len(norms)}, lr {lrs.index(lr)}/{len(lrs)}, loss {losses.index(loss)}/{len(losses)}, optim {optims.index(optim)}/{len(optims)}")
                performaces[f"{optim}_{loss}_{lr}_{n}"] = sum(model_performances) / tests

taken = start-time.time()

with open(args.path+"_metrics.txt", "w") as f:
    f.write(f"Linear: {linear}")
    f.write(f"\nTested: {tests} times")
    f.write(f"\nLoss functions: {' '.join(losses)}")
    f.write(f"\nOptimizers: {' '.join(optims)}")
    f.write(f"\nLearning Rates: {(lrs)}")
    f.write(f"\nNorms: {' '.join(norms)}")
    f.write(f"\nEpochs: {max_epochs}")
    f.write(f"\nBest performances for each setup: {performaces}")
    f.write(f"\nBest performer: {sorted(performaces.items(), key=lambda x:x[1], reverse=True)[0]}")
    f.write(f"\nTook: {math.floor(taken/60)}:{math.floor(taken/60)}:{round(taken%60)}")

print(performaces)
print(f"Best performer: {sorted(performaces.items(), key=lambda x:x[1], reverse=True)[0]}")
print(f"\nTook: {math.floor(taken/3600)}:{math.floor((taken/60)%60):02}:{round(taken%60):02}")