import pandas as pd

def check_tols(train, test, validation, ref):
    entries = ref.groupby(["HC_subCST"]).count()['sampleID']
    total_entries = entries.sum()
    prevelance = entries/total_entries

    # Check data against unsplit data by proportion of each label type
    train_entries = train.groupby(['HC_subCST']).count()['sampleID']
    train_total_entries = train_entries.sum()
    train_prevelance = train_entries/train_total_entries

    test_entries = test.groupby(['HC_subCST']).count()['sampleID']
    test_total_entries = test_entries.sum()
    test_prevelance = test_entries/test_total_entries

    validation_entries = validation.groupby(['HC_subCST']).count()['sampleID']
    validation_total_entries = validation_entries.sum()
    validation_prevelance = validation_entries/validation_total_entries
    print(prevelance,"\n",  validation_prevelance)
    print(len(ref), len(train)+len(test)+len(validation), len(train), len(test), len(validation))

    train_diffs = (abs(train_prevelance / prevelance) - 1) * 100
    test_diffs  = (abs(test_prevelance / prevelance) - 1) * 100
    validation_diffs  = (abs(validation_prevelance / prevelance) - 1) * 100

    worst_train = abs(train_diffs).max()
    worst_test  = abs(test_diffs).max()
    worst_validation  = abs(validation_diffs).max()

    train_diffs = list(map(lambda x:f"{x:2.6f}%", train_diffs)) 
    test_diffs  = list(map(lambda x:f"{x:2.6f}%", test_diffs )) 
    validation_diffs  = list(map(lambda x:f"{x:2.6f}%", validation_diffs ))

    print("Following values are given as decimal (0-1) fequencies in the data set")
    print(f"Prevalances \n Reference: {prevelance} \nTrain: {train_prevelance} \nTest: {test_prevelance} \nValidation: {validation_prevelance}")
    print("Tolerances:")
    print(f"\033[31m\033[1mTrain: {', '.join(train_diffs)} - worst tolerance is {worst_train:2.6f}%", '\033[39m')
    print(f"\033[31m\033[1mTest: {', '.join(test_diffs)} - worst tolerance is {worst_test:2.6f}%", '\033[39m')
    print(f"\033[31m\033[1mValidation: {', '.join(validation_diffs)} - worst tolerance is {worst_validation:2.6f}%", '\033[39m')

if __name__ == "__main__":
    #ravel_path = "../../data/vaginal/France/"
    #train_path = f"{ravel_path}formatted_80_train.csv"
    #test_path = f"{ravel_path}formatted_80_test.csv"
    #validation_path = f"{ravel_path}formatted_80_validation.csv"
    #ref_path = f"{ravel_path}france_formatted.csv"

    # ravel_path = "../../../gut/human_microbiome/"
    # train_path = f"{ravel_path}reigon_data_80_train.csv"
    # test_path = f"{ravel_path}reigon_data_80_test.csv"
    # validation_path = f"{ravel_path}reigon_data_80_validation.csv"
    # ref_path = f"{ravel_path}reigon_processed.csv"

    ravel_path = "../../../gut/proper_enteroypers/"
    train_path = f"{ravel_path}processed_truth_80_train.csv"
    test_path = f"{ravel_path}processed_truth_80_test.csv"
    validation_path = f"{ravel_path}processed_truth_80_validation.csv"
    ref_path = f"{ravel_path}processed_truth.csv"

    #ravel_path = "../../../gut/human_microbiome/"
    #train_path = f"{ravel_path}concordant_set_80_train.csv"
    #test_path = f"{ravel_path}concordant_set_80_test.csv"
    #validation_path = f"{ravel_path}concordant_set_80_validation.csv"
    #ref_path = f"{ravel_path}concordant_set_prcessed.csv"

    #ravel_path = "../../data/oral/Manghi/"
    #train_path = f"{ravel_path}clustered_80_train.csv"
    #test_path = f"{ravel_path}clustered_80_test.csv"
    #validation_path = f"{ravel_path}clustered_80_validation.csv"
    #ref_path = f"{ravel_path}manghi_classified.csv"

    a = pd.read_csv(train_path)
    b = pd.read_csv(test_path)
    c = pd.read_csv(validation_path)
    ref = pd.read_csv(ref_path)

    print("80% training")
    check_tols(a, b, c, ref)
    #exit(1) # Reigon data only has 80%

    #train_path = f"{ravel_path}formatted_60_train.csv"
    #test_path = f"{ravel_path}formatted_60_test.csv"
    #validation_path = f"{ravel_path}formatted_60_validation.csv"

    train_path = f"{ravel_path}processed_truth_60_train.csv"
    test_path = f"{ravel_path}processed_truth_60_test.csv"
    validation_path = f"{ravel_path}processed_truth_60_validation.csv"

    #train_path = f"{ravel_path}concordant_set_60_train.csv"
    #test_path = f"{ravel_path}concordant_set_60_test.csv"
    #validation_path = f"{ravel_path}concordant_set_60_validation.csv"
    
    #train_path = f"{ravel_path}clustered_60_train.csv"
    #test_path = f"{ravel_path}clustered_60_test.csv"
    #test_path = f"{ravel_path}clustered_60_validation.csv"

    a = pd.read_csv(train_path)
    b = pd.read_csv(test_path)
    c = pd.read_csv(validation_path)

    print("60% training")
    check_tols(a, b, c, ref)
