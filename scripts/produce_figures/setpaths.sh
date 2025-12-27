TMP_DIR="tmp_fig_1_and_2"

# Raw VALENCIA data stored in data directory
VALENCIA_DATA="$VALENCIA_REPO/Publication_materials/Data_and_metadata/all_samples_taxonomic_composition_data.csv"

ALL_DATA_FORMATTED="$TMP_DIR/valencia_formatted_full.csv"

# Data used to train 60/20/20 stratabionn model
PREFIX_DATA_60="$TMP_DIR/formatted_60"
TRAIN_DATA_60="${PREFIX_DATA_60}_train.csv"
TEST_DATA_60="${PREFIX_DATA_60}_test.csv"
VALIDATE_DATA_60="${PREFIX_DATA_60}_validation.csv"
VALIDATE_DATA_60_NO_LBL="${PREFIX_DATA_60}_validation_unlabeled.csv"

# Data used to train 80/10/10 stratabionn model
PREFIX_DATA_80="$TMP_DIR/formatted_80"
TRAIN_DATA_80="${PREFIX_DATA_80}_train.csv"
TEST_DATA_80="${PREFIX_DATA_80}_test.csv"
VALIDATE_DATA_80="${PREFIX_DATA_80}_validation.csv"
VALIDATE_DATA_80_NO_LBL="${PREFIX_DATA_80}_validation_unlabeled.csv"

# Generated centroids
VALENCIA_CENTROIDS_STOCK="$VALENCIA_REPO/CST_centroids_012920.csv"
VALENCIA_CENTROIDS_60="$TMP_DIR/centroids_60.csv"
VALENCIA_CENTROIDS_80="$TMP_DIR/centroids_80.csv"

# Classified data paths
CLASSIFIED_STRATABIONN_80="$TMP_DIR/stratabionn_classified_80"
CLASSIFIED_STRATABIONN_60="$TMP_DIR/stratabionn_classified_60"
CLASSIFIED_VALENCIA_60="$TMP_DIR/valencia_classified_60"
CLASSIFIED_VALENCIA_80="$TMP_DIR/valencia_classified_80"
CLASSIFIED_RF_80="$TMP_DIR/rf_classified_80"
CLASSIFIED_RF_60="$TMP_DIR/rf_classified_60"


TMP_DIR="tmp_fig_3"

DATA_DIR="../../data"
ORAL_DATA_DIR="$DATA_DIR/oral"
VAGINAL_DATA_DIR="$DATA_DIR/vaginal"

HICKEY_DATA_DIR="$VAGINAL_DATA_DIR/Hickey"
ASHLEY_DATA_DIR="$ORAL_DATA_DIR/ashley"
EDLUND_DATA_DIR="$ORAL_DATA_DIR/edlund"

HICKEY_DATA="$HICKEY_DATA_DIR/hickey_species-prop.txt"
ASHLEY_DATA="$ASHLEY_DATA_DIR/oral_ashley_2018_species.tab"
EDLUND_DATA="$EDLUND_DATA_DIR/Supplemental_Table_S3_edlund.csv"
RAVEL_DATA="$VALENCIA_DATA"

HICKEY_SCRIPT="$HICKEY_DATA_DIR/hickey_prep.py"
ASHLEY_SCRIPT="$ASHLEY_DATA_DIR/format_ashley.py"
EDLUND_SCRIPT="$EDLUND_DATA_DIR/oral_preprocessor.py"
RAVEL_SCRIPT="../preprocess_valencia.py"

HICKEY_FORMATTED="$TMP_DIR/hickey_formatted.csv"
ASHLEY_FORMATTED="$TMP_DIR/ashley_formatted.csv"
EDLUND_FORMATTED="$TMP_DIR/edlund_formatted.csv"
RAVEL_FORMATTED="$TMP_DIR/ravel_formatted.csv"

OVERLAP_SCRIPT="../find_VAL_overlap.py"
STRATABIONN_SCRIPT="../nn_classifier.py"

TMP_DIR2="tmp_fig_1_and_2"
PREFIX_DATA_80="$TMP_DIR2/formatted_80"
TRAIN_DATA_80="${PREFIX_DATA_80}_train.csv"
TEST_DATA_80="${PREFIX_DATA_80}_test.csv"
VALIDATE_DATA_80="${PREFIX_DATA_80}_validation.csv"
VALIDATE_DATA_80_NO_LBL="${PREFIX_DATA_80}_validation_unlabeled.csv"

MODEL_LOCATION="$TMP_DIR/hickey_classifier"
SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HICKEY_CLASSIFIED="$TMP_DIR/hickey_classified"

EDLUND_PREFIX_DATA_80="$TMP_DIR/formatted_80"
EDLUND_TRAIN_DATA_80="${EDLUND_PREFIX_DATA_80}_train.csv"
EDLUND_TEST_DATA_80="${EDLUND_PREFIX_DATA_80}_test.csv"
EDLUND_VALIDATE_DATA_80="${EDLUND_PREFIX_DATA_80}_validation.csv"

EDLUND_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
EDLUND_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"

TMP_DIR="tmp_fig_3"

# Path to cloned VALENCIA repo root
VALENCIA_REPO="/data/b/class_microbiome/VALENCIA/"

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

MANGHI_TRAIN="$MANGHI_DATA_DIR/clustered_80_train.csv"
MANGHI_TEST="$MANGHI_DATA_DIR/clustered_80_test.csv"
MANGHI_VALIDATE="$MANGHI_DATA_DIR/clustered_80_validation.csv"

MANGHI_MODEL_LOCATION="$TMP_DIR/hickey_classifier"
MANGHI_SIMPLE_MODEL_LOCATION="$TMP_DIR/hickey_classifier_simplified"

HYUHN_CLASSIFIED="$TMP_DIR/hyuhn_classified"

COMMON_COLS_ORAL=$(python3 $OVERLAP_SCRIPT -iv $MANGHI_FORMATTED -i $HYUHN_FORMATTED | tail -n 1)