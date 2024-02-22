ORAL_DATA_PATH="../oral_test_set/Supplemental_Table_S3_edlund.csv"
PROCESSED_DATA_PATH="../oral_test_set/processed_data.csv"
OUTPUT_PATH="../oral_test_set/split"
PREPROCESSOR_SCRIPT="oral_preprocessor.py"
SPLIT_SCIPRT="make_test_train_split.py"

python3 $PREPROCESSOR_SCRIPT -in $ORAL_DATA_PATH -out $PROCESSED_DATA_PATH

python3 $SPLIT_SCIPRT -i $PROCESSED_DATA_PATH -o $OUTPUT_PATH -t 0.03