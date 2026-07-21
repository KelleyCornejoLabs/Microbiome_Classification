#! /bin/bash
# This script produces a bar graph comparing the accuracy, G
# precision, recall, and f1 score of VALENCIA, Stratabionn, 
# and a random forest classifier, corresponding to figure 1.
# It also produces figure 2, which creates confusion matricies
# for VALENCIA, Stratabionn, the random forest classifier,
# and the difference between VALENCIA and Stratabionn


# Ensure we are running from the directory this file is in
# if [[ "$(pwd)" != *"produce_figures"* ]]; then
#     echo "ERR: Please run this script in the produce_figures directory"
#     exit 1
# fi

# Temporary scratch directory
TMP_DIR="models"

# Path to cloned VALENCIA github repo root
# VALENCIA_REPO=""
# VALENCIA_REPO="/data/b/class_microbiome/VALENCIA/"
# if [[ "$VALENCIA_REPO" = "" ]]; then
#     echo "ERR: Path to VALENCIA not specified. Please specify using VALENCIA_REPO"
#     exit 1
# fi

STRATABIONN_PATH="../../Microbiome_Classification/scripts"
TRUTH_DATA="."

# Start with clean empty temp directory
# rm -rf $TMP_DIR
# mkdir $TMP_DIR

ALL_DATA_FORMATTED="$TRUTH_DATA/processed_truth.csv"

# Data used to train 60/20/20 stratabionn model
PREFIX_DATA_60="$TRUTH_DATA/processed_truth_60"
TRAIN_DATA_60="${PREFIX_DATA_60}_train.csv"
TEST_DATA_60="${PREFIX_DATA_60}_test.csv"
VALIDATE_DATA_60="${PREFIX_DATA_60}_validation.csv"

# Data used to train 80/10/10 stratabionn model
PREFIX_DATA_80="$TRUTH_DATA/processed_truth_80" TRAIN_DATA_80="${PREFIX_DATA_80}_train.csv"
TEST_DATA_80="${PREFIX_DATA_80}_test.csv"
VALIDATE_DATA_80="${PREFIX_DATA_80}_validation.csv"

# Paths to store trained Stratabionn models
MODEL_PATH_60="$TMP_DIR/processed_truth_60"
MODEL_PATH_80="$TMP_DIR/processed_truth_80"

# Classified data paths
CLASSIFIED_STRATABIONN_60="$TMP_DIR/stratabionn_classified_60"
CLASSIFIED_STRATABIONN_80="$TMP_DIR/stratabionn_classified_80"
CLASSIFIED_ENTERO_60="$TMP_DIR/entero_classified_60.csv" CLASSIFIED_ENTERO_80="$TMP_DIR/entero_classified_80.csv"
# TODO
#CLASSIFIED_VALENCIA_60="$TRUTH_DATA/entero_classified"

echo "Train a model using Stratabionn and classify validation data"

# Train and classify using 60/20/20 data
# echo "Training for 60%..."
# python3 $STRATABIONN_PATH/nn_classifier.py -itr $TRAIN_DATA_60 -ite $TEST_DATA_60 -p $MODEL_PATH_60 -tm 3 -wip
# echo "Classifying with 60%..."
# python3 $STRATABIONN_PATH/nn_classifier.py -ite $VALIDATE_DATA_60 -p $MODEL_PATH_60 -out $CLASSIFIED_STRATABIONN_60 -cl

# Train and classify using 80/10/10 data
# echo "Training for 80%..."
# python3 $STRATABIONN_PATH/nn_classifier.py -itr $TRAIN_DATA_80 -ite $TEST_DATA_80 -p $MODEL_PATH_80 -tm 3 -wip
# echo "Classifying with 80%..."
# python3 $STRATABIONN_PATH/nn_classifier.py -ite $VALIDATE_DATA_80 -p $MODEL_PATH_80 -out $CLASSIFIED_STRATABIONN_80 -cl

# Put entero stuff here

# Only validate on validation set. Using everything skews results
# Generate figures 8 and 9
# python utilities.py fig_gut_cmp --stratabionn-class-60 $CLASSIFIED_STRATABIONN_60 --stratabionn-class-80 $CLASSIFIED_STRATABIONN_80 \
#                                 --entero-class-60 $CLASSIFIED_ENTERO_60 --entero-class-80 $CLASSIFIED_ENTERO_80 \
#                                 --validation-60 $VALIDATE_DATA_60 --validation-80 $VALIDATE_DATA_80


COMPENDIUM_PROCESSED="human_micro_comp_processed.csv"
COMPENDIUM_CLASSIFIED="human_micro_comp_classified"

OVERLAP_SCRIPT="../../Microbiome_Classification/scripts/find_VAL_overlap.py"

echo "Finding common columns between Enterotyper ground truth and Human Microbiom Compendium data"
COMMON_COLS_GUT=$(python3 $OVERLAP_SCRIPT --files $ALL_DATA_FORMATTED $COMPENDIUM_PROCESSED | tail -n 1)

COMPENDIUM_MODEL_PATH="$TMP_DIR/compendium_model"
COMPENDIUM_SIMPLE_MODEL_PATH="$TMP_DIR/compendium_model_simplified"

#echo "$COMMON_COLS_GUT"

# echo "Training classifier for Human Microbiome Compendium data on Enterotyper ground truth dataset"
# python3 $STRATABIONN_PATH/nn_classifier.py -itr $TRAIN_DATA_80 -ite $TEST_DATA_80 -f $COMMON_COLS_GUT -p $COMPENDIUM_MODEL_PATH -ts -dc -dbg

# echo "Applying classifier to Human Microbiome Compendium dataset"
# python3 $STRATABIONN_PATH/nn_classifier.py -ite $COMPENDIUM_PROCESSED -cl -p $COMPENDIUM_SIMPLE_MODEL_PATH -out $COMPENDIUM_CLASSIFIED -dbg -dc

# This is figure 10
echo "Generating PaCMAP figure"
python utilities.py fig_4 --france-data $ALL_DATA_FORMATTED \
                          --hickey-data $COMPENDIUM_CLASSIFIED.csv \
                          --common_cols $COMMON_COLS_GUT
