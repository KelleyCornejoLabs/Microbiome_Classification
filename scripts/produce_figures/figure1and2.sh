#! /bin/bash
# This script produces a bar graph comparing the accuracy, 
# precision, recall, and f1 score of VALENCIA, Stratabionn, 
# and a random forest classifier, corresponding to figure 1.
# It also produces figure 2, which creates confusion matricies
# for VALENCIA, Stratabionn, the random forest classifier,
# and the difference between VALENCIA and Stratabionn


# Ensure we are running from the directory this file is in
if [[ "$(pwd)" != *"produce_figures"* ]]; then
    echo "ERR: Please run this script in the produce_figures directory"
    exit 1
fi

# Temporary scratch directory
TMP_DIR="tmp_fig_1_and_2"

# Path to cloned VALENCIA github repo root
# VALENCIA_REPO=""
VALENCIA_REPO="/data/b/class_microbiome/VALENCIA/"
if [[ "$VALENCIA_REPO" = "" ]]; then
    echo "ERR: Path to VALENCIA not specified. Please specify using VALENCIA_REPO"
    exit 1
fi

FRANCE_DATA="../../data/vaginal/France"

# Start with clean empty temp directory
# rm -rf $TMP_DIR
# mkdir $TMP_DIR

ALL_DATA_FORMATTED="$FRANCE_DATA/france_formatted_full.csv"

# Data used to train 60/20/20 stratabionn model
PREFIX_DATA_60="$FRANCE_DATA/formatted_60"
TRAIN_DATA_60="${PREFIX_DATA_60}_train.csv"
TEST_DATA_60="${PREFIX_DATA_60}_test.csv"
VALIDATE_DATA_60="${PREFIX_DATA_60}_validation.csv"
VALIDATE_DATA_60_NO_LBL="${PREFIX_DATA_60}_validation_unlabeled.csv"

# Data used to train 80/10/10 stratabionn model
PREFIX_DATA_80="$FRANCE_DATA/formatted_80"
TRAIN_DATA_80="${PREFIX_DATA_80}_train.csv"
TEST_DATA_80="${PREFIX_DATA_80}_test.csv"
VALIDATE_DATA_80="${PREFIX_DATA_80}_validation.csv"
VALIDATE_DATA_80_NO_LBL="${PREFIX_DATA_80}_validation_unlabeled.csv"

# Centroids
VALENCIA_CENTROIDS_STOCK="$VALENCIA_REPO/CST_centroids_012920.csv"
# VALENCIA_CENTROIDS_60="$TMP_DIR/centroids_60.csv"
# VALENCIA_CENTROIDS_80="$TMP_DIR/centroids_80.csv"

# Paths to store trained Stratabionn models
MODEL_PATH_60="$TMP_DIR/classifier_60"
MODEL_PATH_80="$TMP_DIR/classifier_80"

# Classified data paths
CLASSIFIED_STRATABIONN_80="$TMP_DIR/stratabionn_classified_80"
CLASSIFIED_STRATABIONN_60="$TMP_DIR/stratabionn_classified_60"
CLASSIFIED_VALENCIA_60="$TMP_DIR/valencia_classified_60"
CLASSIFIED_VALENCIA_80="$TMP_DIR/valencia_classified_80"
CLASSIFIED_RF_80="$TMP_DIR/rf_classified_80"
CLASSIFIED_RF_60="$TMP_DIR/rf_classified_60"

echo "Train a model using Stratabionn and classify validation data"

# Train and classify using 60/20/20 data
# python3 ../nn_classifier.py -itr $TRAIN_DATA_60 -ite $TEST_DATA_60 -p $MODEL_PATH_60 -tm 3 -wip
# python3 ../nn_classifier.py -ite $VALIDATE_DATA_60 -p $MODEL_PATH_60 -out $CLASSIFIED_STRATABIONN_60 -cl

# Train and classify using 80/10/10 data
# python3 ../nn_classifier.py -itr $TRAIN_DATA_80 -ite $TEST_DATA_80 -p $MODEL_PATH_80 -tm 3 -wip
# python3 ../nn_classifier.py -ite $VALIDATE_DATA_80 -p $MODEL_PATH_80 -out $CLASSIFIED_STRATABIONN_80 -cl

# Classify using Valencia
# echo "Generating VALENCIA centroids"
# python3 ../centroids.py $TRAIN_DATA_60 -o $VALENCIA_CENTROIDS_60 -l HC_subCST -ndc sampleID,read_count
# python3 ../centroids.py $TRAIN_DATA_80 -o $VALENCIA_CENTROIDS_80 -l HC_subCST -ndc sampleID,read_count

# echo "Generating unlabeled validation set for VALENCIA"
# # Drop labels for VALENCIA
# python3 -c "import pandas as pd; pd.read_csv('$VALIDATE_DATA_60').rename(columns={'HC_subCST': 'sub_CST'}).to_csv('$VALIDATE_DATA_60_NO_LBL', index=False)"
# python3 -c "import pandas as pd; pd.read_csv('$VALIDATE_DATA_80').rename(columns={'HC_subCST': 'sub_CST'}).to_csv('$VALIDATE_DATA_80_NO_LBL', index=False)"

# echo "Running VALENCIA"
# python3 $VALENCIA_REPO/Valencia.py -ref $VALENCIA_CENTROIDS_60 -i $VALIDATE_DATA_60_NO_LBL -o $CLASSIFIED_VALENCIA_60
# python3 $VALENCIA_REPO/Valencia.py -ref $VALENCIA_CENTROIDS_80 -i $VALIDATE_DATA_80_NO_LBL -o $CLASSIFIED_VALENCIA_80
# python3 $VALENCIA_REPO/Valencia.py -ref $VALENCIA_CENTROIDS_STOCK -i $VALIDATE_DATA_60_NO_LBL -o $CLASSIFIED_VALENCIA_60
# python3 $VALENCIA_REPO/Valencia.py -ref $VALENCIA_CENTROIDS_STOCK -i $VALIDATE_DATA_80_NO_LBL -o $CLASSIFIED_VALENCIA_80

# Classify using random forest
# echo "Running Random forest classifiers"
# python3 ../random_forest_classifier.py -itr $TRAIN_DATA_60 -ite $VALIDATE_DATA_60 -o $CLASSIFIED_RF_60.csv -dbg
# python3 ../random_forest_classifier.py -itr $TRAIN_DATA_80 -ite $VALIDATE_DATA_80 -o $CLASSIFIED_RF_80.csv -dbg

# TODO: Replace this with better classifications with VALENCIA
# IF we decide it would be better to generate custom centroids. Might be better to use stock
# Classify using Valencia
# python ../preprocess_valencia.py  -i $VALIDATE_DATA_80 -o $VALIDATE_DATA_80_NO_LBL
# python3 $VALENCIA_REPO/Valencia.py -ref $VALENCIA_REPO/CST_centroids_012920.csv -i $VALIDATE_DATA_80_NO_LBL -o $CLASSIFIED_VALENCIA_80

# Only validate on validation set. Using everything skews results
# Generate figures 1 and 2
# python utilities.py fig_1_and_2 --stratabionn-class-60 $CLASSIFIED_STRATABIONN_60 --stratabionn-class-80 $CLASSIFIED_STRATABIONN_80 \
#                                 --forest-class-60 $CLASSIFIED_RF_60 --forest-class-80 $CLASSIFIED_RF_80 \
#                                 --valencia-class-60 $CLASSIFIED_VALENCIA_60 \
#                                 --validation-60 $VALIDATE_DATA_60 --validation-80 $VALIDATE_DATA_80 \
#                                 --valencia-class-80 $CLASSIFIED_VALENCIA_80