#! /bin/bash

# These paths need to be set by user
valencia_path="../VALENCIA/Valencia.py"
valencia_prep_path="./preprocess_valencia.py"
valencia_eval_path="./eval_valencia.py"
test_centroids="./new_centroids_subCST.csv"
train_set_path="./out_train.csv"
test_set_path="./out_test.csv"

# Neural classifier paths also need to be set by user
classifier_path="./nn_classifier.py"
model_path="eval"

# Random forest classifier paths
forest_path="./random_forest_classifier.py"

# These ones are probably fine as is
test_set_prepped_path="./test_processed.csv"
out_path_valencia="valencia_predictions"
out_path_nn="neural_predictions"
out_path_forest="forest_predictions"

# Prepare data and feed to valencia
python $valencia_prep_path -i $test_set_path -o $test_set_prepped_path
python $valencia_path -r $test_centroids -i $test_set_prepped_path -o $out_path_valencia

# Evaluate valencia
python $valencia_eval_path -ip $out_path_valencia.csv -id $test_set_path

# Evaluate neural classifier
python $classifier_path -itr $train_set_path -ite $test_set_path -p $model_path --no-debug
python $classifier_path -ite $test_set_path -cl -lb -out $out_path_nn -p $model_path

python $valencia_eval_path -ip $out_path_nn.csv -id $test_set_path

# Evaluate random forest classifier
python $forest_path --no-debug -itr $train_set_path -ite $test_set_path -o $out_path_forest.csv

python $valencia_eval_path -ip $out_path_forest.csv -id $test_set_path