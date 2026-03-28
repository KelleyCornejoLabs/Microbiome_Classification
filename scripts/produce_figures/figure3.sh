#! /bin/bash
# Produces pacmaps for all datasets,
# France and Hickey for the vaginal,
# and Hyuhm and Baker for the oral,
# corresponding to figure 3

set -eou pipefail

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
BAKER_DATA_DIR="$ORAL_DATA_DIR/Baker"
FRANCE_DATA_DIR="$VAGINAL_DATA_DIR/France"
MANGHI_DATA_DIR="$ORAL_DATA_DIR/Manghi"

HICKEY_FORMATTED="$HICKEY_DATA_DIR/hickey_formatted.csv"
HYUHN_FORMATTED="$HYUHN_DATA_DIR/hyuhn_formatted.csv"
BAKER_FORMATTED="$BAKER_DATA_DIR/baker_formatted.csv"
FRANCE_FORMATTED="$FRANCE_DATA_DIR/france_formatted.csv"
MANGHI_FORMATTED="$MANGHI_DATA_DIR/manghi_classified.csv"

FRANCE_TRAIN="$FRANCE_DATA_DIR/formatted_80_train.csv"
FRANCE_TEST="$FRANCE_DATA_DIR/formatted_80_test.csv"
FRANCE_VALIDATE="$FRANCE_DATA_DIR/formatted_80_validation.csv"

FRANCE_60_TRAIN="$FRANCE_DATA_DIR/formatted_60_train.csv"
FRANCE_60_TEST="$FRANCE_DATA_DIR/formatted_60_test.csv"
FRANCE_60_VALIDATE="$FRANCE_DATA_DIR/formatted_60_validation.csv"

OVERLAP_SCRIPT="../find_VAL_overlap.py"
STRATABIONN_SCRIPT="../nn_classifier.py"

echo "Generating figures 3 and 6"
# python utilities.py --fig new_number_1 fig_3_6 --france-data  $FRANCE_FORMATTED \
#                             --hickey-data $HICKEY_FORMATTED \
#                             --hyuhn-data $HYUHN_FORMATTED \
#                             --baker-data $BAKER_FORMATTED \
#                             --france-train $FRANCE_TRAIN \
#                             --france-test $FRANCE_TEST \
#                             --france-validate $FRANCE_VALIDATE

# Supplemental!
# python utilities.py --fig supl_france_60_pacmap fig_3_6 --france-data  $FRANCE_FORMATTED \
#                             --hickey-data $HICKEY_FORMATTED \
#                             --hyuhn-data $HYUHN_FORMATTED \
#                             --baker-data $BAKER_FORMATTED \
#                             --france-train $FRANCE_60_TRAIN \
#                             --france-test $FRANCE_60_TEST \
#                             --france-validate $FRANCE_60_VALIDATE


echo "Finding common columns between France and Hickey data"
COMMON_COLS=$(python3 $OVERLAP_SCRIPT --files $FRANCE_FORMATTED $HICKEY_FORMATTED | tail -n 1)

# TODO: Merge this with other fig gen script
PREFIX_DATA_80="$FRANCE_DATA_DIR/formatted_80"
TRAIN_DATA_80="${PREFIX_DATA_80}_train.csv"
TEST_DATA_80="${PREFIX_DATA_80}_test.csv"
VALIDATE_DATA_80="${PREFIX_DATA_80}_validation.csv"
VALIDATE_DATA_80_NO_LBL="${PREFIX_DATA_80}_validation_unlabeled.csv"

MODEL_LOCATION="$TMP_DIR/hickey_classifier"
SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HICKEY_CLASSIFIED="$TMP_DIR/hickey_classified"

echo "Training classifier for Hickey data on France dataset"
python3 $STRATABIONN_SCRIPT -itr $TRAIN_DATA_80 -ite $TEST_DATA_80 -f $COMMON_COLS -p $MODEL_LOCATION -ts -tm 3

echo "Applying classifier to Hickey dataset"
python3 $STRATABIONN_SCRIPT -ite $HICKEY_FORMATTED -cl -p $SIMPLE_MODEL_LOCATION -out $HICKEY_CLASSIFIED


# python utilities.py fig_4 --france-data  $FRANCE_FORMATTED \
#                           --hickey-data $HICKEY_CLASSIFIED.csv \
#                           --common_cols $COMMON_COLS

# python utilities.py fig_5 --hyuhn-data $HYUHN_FORMATTED \
#                           --baker-data $BAKER_FORMATTED \
#                           --common_cols $COMMON_COLS

exit 1

# echo "Generating Test/Train/Validation data (60/20/20 and 80/10/10)"

# BAKER_PREFIX_DATA_80="$BAKER_DATA_DIR/formatted_80"
# BAKER_TRAIN_DATA_80="${BAKER_PREFIX_DATA_80}_train.csv"
# BAKER_TEST_DATA_80="${BAKER_PREFIX_DATA_80}_test.csv"
# BAKER_VALIDATE_DATA_80="${BAKER_PREFIX_DATA_80}_validation.csv"

# BAKER_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
# BAKER_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

# HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"

# COMMON_COLS_ORAL=$(python3 $OVERLAP_SCRIPT -iv $BAKER_FORMATTED -i $HYUHN_FORMATTED | tail -n 1)
# echo "Common Cols Oral:"
# echo $COMMON_COLS_ORAL

# echo "Training classifier for Hyuhn data on Baker dataset"
# python3 $STRATABIONN_SCRIPT -itr $BAKER_TRAIN_DATA_80 -ite $BAKER_TEST_DATA_80 -f $COMMON_COLS_ORAL -p $BAKER_MODEL_LOCATION -ts

# echo "Applying classifier to Hyuhn dataset"
# python3 $STRATABIONN_SCRIPT -ite $HYUHN_FORMATTED -cl -p $BAKER_SIMPLE_MODEL_LOCATION -out $HYUHN_CLASSIFIED

# echo "Generating figure 6"

MANGHI_TRAIN="$MANGHI_DATA_DIR/clustered_80_train.csv"
MANGHI_TEST="$MANGHI_DATA_DIR/clustered_80_test.csv"
MANGHI_VALIDATE="$MANGHI_DATA_DIR/clustered_80_validation.csv"

# python utilities.py fig_3_6 --france-data  $FRANCE_FORMATTED \
#                             --hickey-data $HICKEY_FORMATTED \
#                             --hyuhn-data $HYUHN_FORMATTED \
#                             --baker-data $BAKER_FORMATTED \
#                             --france-train $MANGHI_TRAIN \
#                             --france-test $MANGHI_TEST \
#                             --france-validate $MANGHI_VALIDATE

# BAKER_PREFIX_DATA_80="$BAKER_DATA_DIR/formatted_80"
# BAKER_TRAIN_DATA_80="${BAKER_PREFIX_DATA_80}_train.csv"
# BAKER_TEST_DATA_80="${BAKER_PREFIX_DATA_80}_test.csv"
# BAKER_VALIDATE_DATA_80="${BAKER_PREFIX_DATA_80}_validation.csv"

MANGHI_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
MANGHI_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"

# COMMON_COLS_ORAL=$(python3 $OVERLAP_SCRIPT --files $MANGHI_FORMATTED $HYUHN_FORMATTED | tail -n 1)
# echo "Common Cols Oral:"
# echo $COMMON_COLS_ORAL

# echo "Training classifier for Hyuhn data on Baker dataset"
# Probably reduce patience
# python3 $STRATABIONN_SCRIPT -itr $MANGHI_TRAIN -ite $MANGHI_TEST -f $COMMON_COLS_ORAL -p $MANGHI_MODEL_LOCATION -ts

# echo "Applying classifier to Hyuhn dataset"
# python3 $STRATABIONN_SCRIPT -ite $HYUHN_FORMATTED -cl -p $MANGHI_SIMPLE_MODEL_LOCATION -out $HYUHN_CLASSIFIED

# python utilities.py fig_4 --france-data  $MANGHI_FORMATTED \
#                           --hickey-data $HYUHN_CLASSIFIED.csv \
#                           --common_cols $COMMON_COLS_ORAL