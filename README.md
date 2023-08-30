# Microbiome Classification
This repository contains the scripts for the Microbiome Classification project
## Scripts
### preprocess_valencia.py
This script takes the data provided in the [VALENCIA project's repository](https://github.com/ravel-lab/VALENCIA) and preprocesses it into a format that can be used by VALENCIA. All scripts in this repository take .csv files in the same format as VALENCIA.
### make_test_train_split.py
For evaluation of each classification method, data from the VALENCIA project is split into a training and validation set. This tool creates those sets from the preprocessed data.
## nn_classifier.py
This script trains and deploys the neural network classifier for microbiome count data. It creates a file containing the network and a file containing metric data from the training process.
