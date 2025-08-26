#! /bin/bash
# Produces pacmaps for all datasets,
# VALENCIA and hikcey for the vaginal,
# and ashley and edlund for the oral,
# corresponding to figure 3

# Ensure we are running from the directory this file is in
if [[ "$(pwd)" != *"produce_figures"* ]]; then
    echo "ERR: Please run this script in the produce_figures directory"
    exit 1
fi

# Temporary scratch directory
TMP_DIR="tmp_fig_3"

# Start with clean empty temp directory
rm -rf $TMP_DIR
mkdir $TMP_DIR

# Path to cloned VALENCIA repo root
VALENCIA_REPO="/data/b/class_microbiome/VALENCIA/"
if [[ "$VALENCIA_REPO" = "" ]]; then
    echo "ERR: Path to VALENCIA not specified. Please specify using VALENCIA_REPO"
    exit 1
fi

# Raw VALENCIA data stored in data directory
VALENCIA_DATA="$VALENCIA_REPO/Publication_materials/Data_and_metadata/all_samples_taxonomic_composition_data.csv"

# Check if data file exists
if [ ! -f $VALENCIA_DATA ]; then
    echo "VALENCIA all_samples_taxononmic_composition_data.csv not found in the following path:"
    echo "$VALENCIA_DATA"
    exit 1
fi

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

HICKEY_FORMATTED="$TMP_DIR/hickey_Formatted.csv"
ASHLEY_FORMATTED="$TMP_DIR/ashley_formatted.csv"
EDLUND_FORMATTED="$TMP_DIR/edlund_formatted.csv"
RAVEL_FORMATTED="$TMP_DIR/ravel_formatted.csv"

# Format vaginal data
echo "Processing Ravel data"
python3 $RAVEL_SCRIPT -i $RAVEL_DATA -o $RAVEL_FORMATTED -kl

echo "Processing Hickey data"
python3 $HICKEY_SCRIPT -i $HICKEY_DATA -o $HICKEY_FORMATTED

# Format oral data
# TODO: Set randomness seed?
echo "Processing Edlund data"
python3 $EDLUND_SCRIPT -in $EDLUND_DATA -out $EDLUND_FORMATTED -cl 3 --graph

echo "Processing Ashley data"
python3 $ASHLEY_SCRIPT -i $ASHLEY_DATA -o $ASHLEY_FORMATTED

python utilities.py fig_3 --ravel-data  $RAVEL_FORMATTED \
                          --hickey-data $HICKEY_FORMATTED \
                          --ashley-data $ASHLEY_FORMATTED \
                          --edlund-data $EDLUND_FORMATTED