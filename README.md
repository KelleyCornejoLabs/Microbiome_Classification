# Microbiome Classification
This repository contains the scripts for the Microbiome Classification project

![Logo](logo.png)

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

## Procedure
This section will outline the basic procedure to use StrataBionn to classify labeled data from the dataset included in the Valencia repository, and evlauate the accuracy of these classifications. This process will be the same for any other dataset used, with notes made where the process would be differ.

### Splitting the dataset
*Note: The dataset provided in the Valencia github repository requires some pre-processing before it can be used with the Valencia tool. StrataBionn uses the same dataset formatting as Valencia, so some pre-processing must be done in order to use this dataset. This preprocessing can be done either by the included `preprocess_valencia.py` script, or it can be done automatically by `make_test_train_split.py` before splitting. `make_test_train_split.py` must be manually configured to process datasets which use a different formatting.* 

We will split the data into 3 different sets for training, testing, and validating the model. The training set will be used to discover the patterns used to make classifications, the test set will be used in the training process to detect overfitting, and the validation set will be used to check the final accuracy of the model on never before seen data. We will split the dataset into an 80/10/10 training/test/validaiton split. The default options for the make_test_train_split script are set to accept the valencia file, so we don't need to set any of them. Other datasets may require some options to be manually set (options are described lower in this readme).

`python3 /this_repo/scripts/make_test_train_split.py -i /Valencia/all_samples_taxonomic_composition_data.csv -s 80 -v 10 -o "valencia_data" -t 0.0015`

This will create files called valencia_data_train.csv, valencia_data_test.csv, and valencia_data_validation.csv. The tolerance is 0.0005 above the default allowed tolerance (0.001 or 0.1%) so it needs to be increased. The tolerance corresponds to the maximum percent difference in class distributions across the split datasets.

### Training a classifier
Now we can train a model on our split Valencia dataset. We will use the valencia_data_train.csv set to find patterns, and we will use the valencia_data_test.csv set to make sure the model is not overfitting to the training set. The test set is also used to measure the accuracy on "unseen" data during training. We can use the following command to train using the default settings, and save the classifier model under the name 'saved_model' (the file containing model will be saved_model_nn.pt, but it can be loaded by entering 'saved_model').

`python3 nn_classifier.py -ite valencia_data_test.csv -itr valencia_data_train.csv -p saved_model`

At the end, three plots will appear. One shows the loss over time, and another shows a confusion matrix, and the last compares sample classification based on the selected species's count data. These graphs will be explained in greater detail in the next section. They can be closed by pressing the q key or closing the window as normal.

### Testing a model
You may have seen some of the stats printed during training. If you want to see these again, or see the stats for a different dataset, run the following command:

`python3 nn_classifier.py -ite valencia_data_validation.csv -p saved_model -ta`

Notice the above command uses the validation set, which the model has not seen yet. This is important because the model looks for patterns found within the training data, and stopping conditions are set based on performance how well the patterns work on the test set. To know if these patterns are general enough to work on any data, we need to use unseen data to validate them.

[//]: # (the model keeps running until it finds patterns that work for the test set.)

After running this command, the accuracy will be printed in the console, and the confusion matrix will appear. The numbers show how many samples with the corresponding true label were classified as the corresponding predicted label type. A heatmap is also provided to highlight hotspots where the model may be making a large number of incorrect assignments. Correct assignments appear in a diagonal line from the top left to the bottom right. After closing the confusion matrix a text based representation of it will be printed to the terminal, and the weighted F1 score will be printed. The next graph to appear will plot all samples on a 2D graph based on their count data for two different bacteria species. The samples are color coded based on their assigned CST to help visualize the decision boundaries that make the model work. The buttons allow you to select which two species to compare, and you can cycle through them in alphabetical order. After closing this interactive graph, the cumulative guess rankings are printed to the terminal. These show how many of the classifier's first predictions were correct, followed by the number of predictions where the model's second guess would have been correct, continuing until the last guess. If the model's second guess is frequently correct, that suggests there may be two very similar calssifications which may be very difficult to differentiate.

## Neural Classifier usage
The neural Classifier is the most promising and feature rich classification method. It should be capable of creating a classifier for any labeled set of training data for use on any unlabeled set of data.

### Input files
The neural classifier can accept three types of input data:
1. Training data - This labeled data is used to train a new neural classifier, and is what the classifier looks at to find patterns in features
1. Testing data (labeled) - This data is used to evaluate the model when training to see how well it generalizes, and is used to evaluate accuracy of the model
1. Testing data (unlabeled) - This is data formatted the same as the data above, but without any labeles. This is used in conjunction with a trained model to generate predictions for the label of that data.

The data is expected as a csv file with rows as samples, and columns as features

The data is expected to have columns named sampleID and read_count, and labeled data is expected to have a column HC_subCST for the label. These names are borrowed from the VALENCIA program's column names, and options are provided to use different names fore each column.

### Common options
The most useful options are: \
`--input-train` / `-itr` - This option is used to provide the path to the labeled training data file (a csv as described above)\
`--input-test` / `-ite` - This option is used to provide the path to the test data, either labeled or unlabeled depending on other options\
`--path` / `-p` - This options is used to provide a path to where the model and any diagnostic data should be saved. It is a prefix used for the model (\*_nn.pt) and the metric data (\*_metrics.txt) and the loss curves (\*_plt.png)\
`--classify` / `-cl` - This option puts the program into classify mode (training mode is default) where the test data is interpreted as unlabeled data, and classifications are produced based on the trained model provided\
`--output` / `-out` - This options is used to provide a path to where a csv with classifications for each sample should be produced

These options are enough to train and use a simple model with all the other settings as default. An example would be:
`python3 nn_classifier.py --input-train nn_training_train.csv --input-test nn_training_test.csv --path models/model1` To train a model on nn_training_train and using nn_training_test for accuracy scoring, and put the trained model in models/model1_nn.pt
`python3 nn_classifier.py --input-test nn_testing_unlabeled.csv --path models/model1 --classify` This command puts the program into classify mode, and uses the model trained in the previous command to evaluate the unlabeled data with the same features as the training data 

### All options
Other than the previously mentioned options, there are more options available. All of these have sensible defaults, but can be tuned to work better for your use case

Boolean options to enable/disable a feature/mode should be used when you want to turn the feature/mode on. Turning off a feature is done by adding `no-` to the beginning. For example to turn of debug output (enabled by default) you would use `--no-debug`.

`--threshold-lr` / `-tlr` - This is the training cuttoff point for the model. It uses learning rate scheduling to drop the learning rate when it is stagnant, and this command specifies the threshold to terminate training\
`--learning-rate` / `-lr` - This option sets the starting learning rate for the neural network\
`--max-epochs` / `-me` - This option sets a hard limit on the number of epochs to train for, and will terminate training if the stopping threshold isn't reached before then\
`--train` / `-t` - Puts the program in training mode. Redundant as this is the default\
`--continue-train` / `-c` - Puts the program into a mode where it can continue training a previously trained model. Not usually reccomended\
`--metrics-interval` / `-m` - After how many epochs the model should print metrics info to console and log file\
`--loss` / `-l` - Sets the loss function for the model. Options are ce (cross entropy - default), nll (negative log likelyhood), or kld (Kullback-Leibler divergence)\
`--load` / `-lo` - Load a fully trained model in order to train a simpler one\
`--optim` / `-o` - The optimizer to use when training a nerural classifier. Options are sgd (stochastic gradient descent) and adam\
`--linear` / `-li` - Trains a linear model as opposed to the default non-linear model with ReLU activation\
`--patience` / `-pa` - How many epochs of no or negative improvement in accuracy before the learning rate is lowered\
`--seed` / `-sd` - Seed the random number generators for more reproducable results\
`--hidden-neurons` / `-hn` - Number of hidden neurons to use in the hidden layer. Default is (2/3)*in_featres + classes\
`--debug` / `-dbg` - Print debug/metrics information. On by defualt, can be disabled with `--no-debug`\
`--train-simple` / `-ts` - Enables trianing a simpler model based on either a newly trained model, or the model from `--path` when `--load` is used. Highly reccomended to be used in conjunction with `--train-multiple` for the best results\
`--test-accuracy` / `-ta` - Given test input data and a model path, evalutate the models accuracy and print results\
`--train-multiple` / `-tm` - Trains the given number of models, and picks the best. Default is 1\
`--info` / `-i` - Parse the model provided with `--path` and print info about it\
`--focus-columns` / `-fc` - Force simpler models to focus on these columns, ignoring the default importance evaluations\
`--labeled` / `-lb` - Is the data provided for classification mode labeled\
`--normalizing-function` / `-n` - Function used to normalize the data before training. All methods normalize by the total count first to account for sample quality differences. Default reccomended\
`--regex-remove` / `-rr` - When loading data use this regex to remove features\
`--dropout` / `-dr` - Sets the dropout parameter. The droupout layer is between the two linear layers\
`--importance-thresh` / `-it` - Sets the minimum affect on accuracy a feature must have for it to be considered important when training a simpler network (percent, default is 0.5% (`--importance-thresh 0.5`))
