# StrataBioNN Microbiome Classification
This repository contains the scripts for the Microbiome Classification project

![Logo](logo.png)

## Main Scripts
### nn_classifier.py
This script trains and deploys the neural network classifier for microbiome count data. It creates a file containing the network and a file containing metric data from the training process. For training, it takes a training set and test set as input. Use make_test_train_split.py for making the split. For comparing different model's performance, split the data once to reserve a validation set, then split the training data again to create the training/test sets for creating models.
### random_forect_classifier.py
This script trains a random forest classifier on the given training data and evaluates is accuracy based on the testing data.

## Utility Scripts
These scripts can be found in the utilities subdirectory
### centroids.py
This script can be used to generate a file containing CST centroids for use with the VALENCIA classifier
### check_tolerances.py
Checks the distribution of class labels for train/test/validation subsets against original dataset to report their tolerance. Tolerance is calaculated as `|(p_class_sub/p_class_super)-1|*100` where `p_class_sub` is the proportion of entries in a subset labeled as class X, while `p_class_super` is the proportion of entries in the superset labeled as class X. This analysis is performed for each subset (train, test, and validation)
### eval_valencia.py
Not necessarily just for valencia. Takes the output file of the VALENCIA or neural classifier, and shows a confusion matrix and accuracy score for the classifiers.
### evaluate_valencia.sh
Shell script that displays the accuracy of VALENCIA's predictions based on the file it was trained on, comparing it to the HC_subCST label for the heirarchical clustering results.
### find_VAL_overlap.py
Takes a list of files as an input. Used to find list of bacteria that match across studies to train 'simple' model for cross-study analysis
### make_test_train_split.py
For evaluation of each classification method, data from the VALENCIA project is split into a training and validation set. This tool creates those sets from the preprocessed data.
### nn_trainer.py
This script trains multiple classifiers, and finds the most performant hyperperameters from those provided to search.
### oral_preprocessor.py
Reformats data from [Manghi et al.](https://www.nature.com/articles/s41467-024-53934-7), and applies a novel K-means clustering based classification scheme to produce baseline oral microbiome dataset.
### preprocess_valencia.py
This script takes the data provided in the [VALENCIA project's repository](https://github.com/ravel-lab/VALENCIA) and preprocesses it into a format that can be used by VALENCIA. All scripts in this repository take .csv files in the same format as VALENCIA.
### process_oral.sh
This script runs the oral microbiome data preprocessor script, and then splits the resulting processed data into a training, testing, and validation subset.
### test_multiple.sh
Does the same as `evaluate_valencia.sh` but on all models, and shows an accuracy score and confusion matrix for each

## Procedure for training on VALENCIA repository data
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

## Procedure for applying VALENCIA classifications to an unlabeled vaginal microbiome dataset
This section will outline the procedure used to classify samples from a different dataset, which in our case will the the ??? dataset from Hickey et al. The general procedure will be training a simple model on the species found in both datasets, labeling the unlabeled data to create a dataset, and finally training a new model on the newly labeled dataset. This procedure will be applicable when applying any pre-existing classification onto an unclassified dataset, VALENCIA is just used for ease of example. The unclassified datasets will need to use a VALENCIA/StrataBionn compliant formatting. How to reformat the data will be described in detail later.

### Training a simple model for both datasets
In order to apply the valencia classifications to our new data, we first need to find the overlap in these datasets, or the bacteria species which are found in both datasets. For this purpose we created the tool `find_VAL_overlap.py` found in this repository. Provided both datasets it will use the classifiers string normalization function to find species which are "the same". You can run the command below to get the common species.

`python3 find_VAL_overlap.py --files valencia_train.csv hickey_train.csv`

This will produce a line saying `--For use in nn_classifier:--` followed by a comma seperated list of the the common bacteria.

We now need to train a 'simple' model on the VALENCIA dataset using only these common bacteria. This is a feature of the neural network classifier, and the features can be specified by using the focus-columns argument. The below command can be used specifically for classifying the hickey data using the VALENCIA classifier.

`python3 nn_classifier.py -ite valencia_test.csv -itr valencia_train.csv -p simple_model -tm 3 --no-train -ts -fc acidaminococcus,acinetobacter,actinobaculum,actinomyces,aerococcus,akkermansia,alistipes,alloscardovia,anaerococcus,anaeroglobus,anaerostipes,arcanobacterium,bacillus,bacteroides,bergeyella,betaproteobacteria,bifidobacterium,blautia,brevibacterium,bulleidia,campylobacter,carnobacteriaceae,catonella,citrobacter,clostridia,clostridiales,collinsella,corynebacteriaceae,corynebacterium,cupriavidus,delftia,dermabacter,dialister,dorea,enhydrobacter,enterobacter,enterobacteriaceae,enterococcus,facklamia,faecalibacterium,fastidiosipila,finegoldia,fusobacterium,gallicola,gardnerella_vaginalis,gemella,globicatella,granulicatella,haemophilus,helcobacillus,helcococcus,howardella,jonquetella,klebsiella,kocuria,lachnospira,lachnospiraceae,lactobacillus_coleohominis,lactobacillus_delbrueckii,lactobacillus_fermentum,lactobacillus_iners,lactobacillus_jensenii,lactobacillus_reuteri,lactobacillus_salivarius,lactococcus,leptotrichia,leucobacter,megasphaera,micrococcus,mobiluncus,mogibacterium,moryella,murdochiella,mycoplasma,negativicoccus,neisseria,neisseriaceae,nosocomiicoccus,oligella,olsenella,parabacteroides,parvimonas,pasteurellaceae,peptococcus,peptoniphilus,peptostreptococcaceae,peptostreptococcus,phascolarctobacterium,porphyromonas,prevotella,prevotellaceae,propionibacterium,propionimicrobium,proteobacteria,pyramidobacter,ralstonia,roseburia,rothia,ruminococcaceae,sarcina,serratia,shewanella,slackia,sneathia,solobacterium,sphingomonas,staphylococcus,stenotrophomonas,subdoligranulum,sutterella,tessaracoccus,trueperella,ureaplasma,varibaculum,variovorax,veillonella,weeksella,zimmermannella`

The flags used here can be a bit confusing, as the 'train simpler model' feature was developed to run after first training a model on all a datasets features. In this context it is used with the `--no-train` flag, which skips the initial model training on all a datasets features. This naming convention is a bit confusing, and should be changed. The train multiple `-tm` flag is set to 3, as simpler models tend to be much more susceptible to bad random starting conditions as there is less data to discover useful features. The focus columns `-fc` flag is set to the bacteria listed from the previous command, which allows the model to be used on either dataset. The focus columns flag only applies to the simpler model

NOTE: Change --no-train flag name. Why does focus columns only apply to simpler model?

### Labeling the unlabeled data to create a dataset
The classifier has a feature to classify an input data file based on a previously trained classifier. The below command takes in the hickey unlabeled data and uses our previously trained simple model, and produces classified file which will become our dataset.

`python3 nn_classifier.py -ite hickey.csv -cl -out hickey_classified -p simple_model`

Now that we have labeled data, we can use the test/train split script to generate a test and training set. The output of the classifer does have some non-data columns in addition to the labels, which are percentage confidence values for each class. The below command removes the non-data values and formats the data for use with the classifier.

`python3 make_test_train_split.py -i hickey_classified.csv -o hickey_dataset -sid sampleID -rc read_count -lc subCST -nd "Pct I-A,Pct I-B,Pct II,Pct III-A,Pct III-B,Pct IV-A,Pct IV-B,Pct IV-C0,Pct IV-C1,Pct IV-C2,Pct IV-C3,Pct IV-C4,Pct V" -t 0.01 -s 80`

We can pass these percentage values to the data splitter script as non-data (-nd) to automatically remove them for us, and use the -lc command

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
