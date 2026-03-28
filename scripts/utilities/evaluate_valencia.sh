#! /bin/bash

# These paths need to be set by user
valencia_path="../VALENCIA/Valencia.py"
valencia_prep_path="./preprocess_valencia.py"
valencia_eval_path="./eval_valencia.py"
#test_centroids="./new_centroids_subCST.csv"
test_centroids="../VALENCIA/CST_centroids_012920.csv"
test_set_path="./out_test.csv"

# These ones are probably fine as is
test_set_prepped_path="./test_processed.csv"
out_path="valencia_predictions"

# Prepare data and feed to valencia
python $valencia_prep_path -i $test_set_path -o $test_set_prepped_path
python $valencia_path -r $test_centroids -i $test_set_prepped_path -o $out_path

# Evaluate valencia
python $valencia_eval_path -ip $out_path.csv -id $test_set_path