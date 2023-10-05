# Microbiome Classification
This repository contains the scripts for the Microbiome Classification project
## Scripts
### preprocess_valencia.py
This script takes the data provided in the [VALENCIA project's repository](https://github.com/ravel-lab/VALENCIA) and preprocesses it into a format that can be used by VALENCIA. All scripts in this repository take .csv files in the same format as VALENCIA.
### make_test_train_split.py
For evaluation of each classification method, data from the VALENCIA project is split into a training and validation set. This tool creates those sets from the preprocessed data.
### nn_classifier.py
This script trains and deploys the neural network classifier for microbiome count data. It creates a file containing the network and a file containing metric data from the training process. For training, it takes a training set and test set as input. Use make_test_train_split.py for making the split. For comparing different model's performance, split the data once to reserve a validation set, then split the training data again to create the training/test sets for creating models.
### nn_trainer.py
This script trains multiple classifiers, and finds the most performant hyperperameters from those provided to search.
### random_forect_classifier.py
This script trains a random forest classifier on the given training data and evaluates is accuracy based on the testing data.
### eval_valencia.py
Not necessarily just for valencia. Takes the output file of the VALENCIA or neural classifier, and shows a confusion matrix and accuracy score for the classifiers.
### evaluate_valencia.sh
Shell script that displays the accuracy of VALENCIA's predictions based on the file it was trained on, comparing it to the HC_subCST label for the heirarchical clustering results.
### test_multiple.sh
Does the same as `evaluate_valencia.sh` but on all models, and shows an accuracy score and confusion matrix for each
