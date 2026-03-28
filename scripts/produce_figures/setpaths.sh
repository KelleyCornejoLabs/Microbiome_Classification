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

BAKER_PREFIX_DATA_80="$BAKER_DATA_DIR/formatted_80"
BAKER_TRAIN_DATA_80="${BAKER_PREFIX_DATA_80}_train.csv"
BAKER_TEST_DATA_80="${BAKER_PREFIX_DATA_80}_test.csv"
BAKER_VALIDATE_DATA_80="${BAKER_PREFIX_DATA_80}_validation.csv"

BAKER_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
BAKER_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"

COMMON_COLS_ORAL=$(python3 $OVERLAP_SCRIPT -iv $BAKER_FORMATTED -i $HYUHN_FORMATTED | tail -n 1)

MANGHI_TRAIN="$MANGHI_DATA_DIR/clustered_80_train.csv"
MANGHI_TEST="$MANGHI_DATA_DIR/clustered_80_test.csv"
MANGHI_VALIDATE="$MANGHI_DATA_DIR/clustered_80_validation.csv"

MANGHI_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
MANGHI_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"