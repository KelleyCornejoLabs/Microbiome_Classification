#! /bin/bash
# Produces pacmaps for all datasets,
# Ravel and Hickey for the vaginal,
# and Hyuhm and Edlund for the oral,
# corresponding to figure 3

# Ensure we are running from the directory this file is in
if [[ "$(pwd)" != *"produce_figures"* ]]; then
    echo "ERR: Please run this script in the produce_figures directory"
    exit 1
fi

# Temporary scratch directory
TMP_DIR="tmp_fig_3"

# Start with clean empty temp directory
# rm -rf $TMP_DIR
# mkdir $TMP_DIR

# Path to cloned VALENCIA repo root
VALENCIA_REPO="/data/b/class_microbiome/VALENCIA/"
if [[ "$VALENCIA_REPO" = "" ]]; then
    echo "ERR: Path to VALENCIA not specified. Please specify using VALENCIA_REPO"
    exit 1
fi

DATA_DIR="../../data"
ORAL_DATA_DIR="$DATA_DIR/oral"
VAGINAL_DATA_DIR="$DATA_DIR/vaginal"

HICKEY_DATA_DIR="$VAGINAL_DATA_DIR/Hickey"
HYUHN_DATA_DIR="$ORAL_DATA_DIR/Hyuhn"
EDLUND_DATA_DIR="$ORAL_DATA_DIR/Edlund"
RAVEL_DATA_DIR="$VAGINAL_DATA_DIR/Ravel"
MANGHI_DATA_DIR="$ORAL_DATA_DIR/Manghi"

HICKEY_FORMATTED="$HICKEY_DATA_DIR/hickey_formatted.csv"
HYUHN_FORMATTED="$HYUHN_DATA_DIR/hyuhn_formatted.csv"
EDLUND_FORMATTED="$EDLUND_DATA_DIR/edlund_formatted.csv"
RAVEL_FORMATTED="$RAVEL_DATA_DIR/ravel_formatted.csv"
MANGHI_FORMATTED="$MANGHI_DATA_DIR/manghi_classified.csv"

RAVEL_TRAIN="$RAVEL_DATA_DIR/formatted_80_train.csv"
RAVEL_TEST="$RAVEL_DATA_DIR/formatted_80_test.csv"
RAVEL_VALIDATE="$RAVEL_DATA_DIR/formatted_80_validation.csv"

OVERLAP_SCRIPT="../find_VAL_overlap.py"
STRATABIONN_SCRIPT="../nn_classifier.py"

echo "Generating figures 3 and 6"
python utilities.py fig_3_6 --ravel-data  $RAVEL_FORMATTED \
                            --hickey-data $HICKEY_FORMATTED \
                            --hyuhn-data $HYUHN_FORMATTED \
                            --edlund-data $EDLUND_FORMATTED \
                            --ravel-train $RAVEL_TRAIN \
                            --ravel-test $RAVEL_TEST \
                            --ravel-validate $RAVEL_VALIDATE


echo "Finding common columns between Ravel and Hickey data"
COMMON_COLS=$(python3 $OVERLAP_SCRIPT -iv $RAVEL_FORMATTED -i $HICKEY_FORMATTED | tail -n 1)

# TODO: Merge this with other fig gen script
PREFIX_DATA_80="$RAVEL_DATA_DIR/formatted_80"
TRAIN_DATA_80="${PREFIX_DATA_80}_train.csv"
TEST_DATA_80="${PREFIX_DATA_80}_test.csv"
VALIDATE_DATA_80="${PREFIX_DATA_80}_validation.csv"
VALIDATE_DATA_80_NO_LBL="${PREFIX_DATA_80}_validation_unlabeled.csv"

MODEL_LOCATION="$TMP_DIR/hickey_classifier"
SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HICKEY_CLASSIFIED="$TMP_DIR/hickey_classified"

echo "Training classifier for Hickey data on Ravel dataset"
# python3 $STRATABIONN_SCRIPT -itr $TRAIN_DATA_80 -ite $TEST_DATA_80 -f $COMMON_COLS -p $MODEL_LOCATION -ts -tm 3

echo "Applying classifier to Hickey dataset"
# python3 $STRATABIONN_SCRIPT -ite $HICKEY_FORMATTED -cl -p $SIMPLE_MODEL_LOCATION -out $HICKEY_CLASSIFIED


python utilities.py fig_4 --ravel-data  $RAVEL_FORMATTED \
                          --hickey-data $HICKEY_CLASSIFIED.csv \
                          --common_cols $COMMON_COLS

# python utilities.py fig_5 --hyuhn-data $HYUHN_FORMATTED \
#                           --edlund-data $EDLUND_FORMATTED \
#                           --common_cols $COMMON_COLS


# echo "Generating Test/Train/Validation data (60/20/20 and 80/10/10)"

# EDLUND_PREFIX_DATA_80="$EDLUND_DATA_DIR/formatted_80"
# EDLUND_TRAIN_DATA_80="${EDLUND_PREFIX_DATA_80}_train.csv"
# EDLUND_TEST_DATA_80="${EDLUND_PREFIX_DATA_80}_test.csv"
# EDLUND_VALIDATE_DATA_80="${EDLUND_PREFIX_DATA_80}_validation.csv"

# EDLUND_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
# EDLUND_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

# HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"

# COMMON_COLS_ORAL=$(python3 $OVERLAP_SCRIPT -iv $EDLUND_FORMATTED -i $HYUHN_FORMATTED | tail -n 1)
# echo "Common Cols Oral:"
# echo $COMMON_COLS_ORAL

# echo "Training classifier for Hyuhn data on Edlund dataset"
# python3 $STRATABIONN_SCRIPT -itr $EDLUND_TRAIN_DATA_80 -ite $EDLUND_TEST_DATA_80 -f $COMMON_COLS_ORAL -p $EDLUND_MODEL_LOCATION -ts

# echo "Applying classifier to Hyuhn dataset"
# python3 $STRATABIONN_SCRIPT -ite $HYUHN_FORMATTED -cl -p $EDLUND_SIMPLE_MODEL_LOCATION -out $HYUHN_CLASSIFIED

echo "Generating figure 6"

MANGHI_TRAIN="$MANGHI_DATA_DIR/clustered_80_train.csv"
MANGHI_TEST="$MANGHI_DATA_DIR/clustered_80_test.csv"
MANGHI_VALIDATE="$MANGHI_DATA_DIR/clustered_80_validation.csv"

python utilities.py fig_3_6 --ravel-data  $RAVEL_FORMATTED \
                            --hickey-data $HICKEY_FORMATTED \
                            --hyuhn-data $HYUHN_FORMATTED \
                            --edlund-data $EDLUND_FORMATTED \
                            --ravel-train $MANGHI_TRAIN \
                            --ravel-test $MANGHI_TEST \
                            --ravel-validate $MANGHI_VALIDATE

# EDLUND_PREFIX_DATA_80="$EDLUND_DATA_DIR/formatted_80"
# EDLUND_TRAIN_DATA_80="${EDLUND_PREFIX_DATA_80}_train.csv"
# EDLUND_TEST_DATA_80="${EDLUND_PREFIX_DATA_80}_test.csv"
# EDLUND_VALIDATE_DATA_80="${EDLUND_PREFIX_DATA_80}_validation.csv"

MANGHI_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
MANGHI_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"

COMMON_COLS_ORAL=$(python3 $OVERLAP_SCRIPT -iv $MANGHI_FORMATTED -i $HYUHN_FORMATTED | tail -n 1)
echo "Common Cols Oral:"
echo $COMMON_COLS_ORAL

# echo "Training classifier for Hyuhn data on Edlund dataset"
# Probably reduce patience
# python3 $STRATABIONN_SCRIPT -itr $MANGHI_TRAIN -ite $MANGHI_TEST -f $COMMON_COLS_ORAL -p $MANGHI_MODEL_LOCATION -ts

# echo "Applying classifier to Hyuhn dataset"
# python3 $STRATABIONN_SCRIPT -ite $HYUHN_FORMATTED -cl -p $MANGHI_SIMPLE_MODEL_LOCATION -out $HYUHN_CLASSIFIED

python utilities.py fig_4 --ravel-data  $MANGHI_FORMATTED \
                          --hickey-data $HYUHN_CLASSIFIED.csv \
                          --common_cols $COMMON_COLS_ORAL