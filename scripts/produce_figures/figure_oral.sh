#! /bin/bash

# Ensure we are running from the directory this file is in
if [[ "$(pwd)" != *"produce_figures"* ]]; then
    echo "ERR: Please run this script in the produce_figures directory"
    exit 1
fi

TMP_DIR="tmp_oral"

# rm -rf $TMP_DIR
# mkdir $TMP_DIR

MANGHI_DATA="../../data/oral/Manghi"

# Data used to train 60/20/20 stratabionn model
PREFIX_DATA_60="$MANGHI_DATA/clustered_60"
TRAIN_DATA_60="${PREFIX_DATA_60}_train.csv"
TEST_DATA_60="${PREFIX_DATA_60}_test.csv"
VALIDATE_DATA_60="${PREFIX_DATA_60}_validation.csv"
VALIDATE_DATA_60_NO_LBL="${PREFIX_DATA_60}_validation_unlabeled.csv"

# Data used to train 80/10/10 stratabionn model
PREFIX_DATA_80="$MANGHI_DATA/clustered_80"
TRAIN_DATA_80="${PREFIX_DATA_80}_train.csv"
TEST_DATA_80="${PREFIX_DATA_80}_test.csv"
VALIDATE_DATA_80="${PREFIX_DATA_80}_validation.csv"
VALIDATE_DATA_80_NO_LBL="${PREFIX_DATA_80}_validation_unlabeled.csv"

# Paths to store trained Stratabionn models
ORAL_MODEL_PATH_60="$TMP_DIR/oral_classifier_60"
ORAL_MODEL_PATH_80="$TMP_DIR/oral_classifier_80"

# Classified data paths
CLASSIFIED_STRATABIONN_80="$TMP_DIR/oral_classified_80"
CLASSIFIED_STRATABIONN_60="$TMP_DIR/oral_classified_60"

# # Train and classify using 60/20/20 data
# python3 ../nn_classifier.py -itr $TRAIN_DATA_60 -ite $TEST_DATA_60 -p $ORAL_MODEL_PATH_60 -tm 3 -wip -hn 150
# python3 ../nn_classifier.py -ite $VALIDATE_DATA_60 -p $ORAL_MODEL_PATH_60 -out $CLASSIFIED_STRATABIONN_60 -cl

# # Train and classify using 80/10/10 data
# python3 ../nn_classifier.py -itr $TRAIN_DATA_80 -ite $TEST_DATA_80 -p $ORAL_MODEL_PATH_80 -tm 3 -wip -hn 150
# python3 ../nn_classifier.py -ite $VALIDATE_DATA_80 -p $ORAL_MODEL_PATH_80 -out $CLASSIFIED_STRATABIONN_80 -cl

# Produce Figure 6 and 7
python utilities.py fig_6 --stratabionn-class-60 $CLASSIFIED_STRATABIONN_60.csv --stratabionn-class-80 $CLASSIFIED_STRATABIONN_80.csv \
                                --validation-60 $VALIDATE_DATA_60 --validation-80 $VALIDATE_DATA_80

# Move this script into this repo
python ./../../../scripts/pacmap_graph.py -nd sampleID -i $MANGHI_DATA/manghi_classified.csv -o fig5