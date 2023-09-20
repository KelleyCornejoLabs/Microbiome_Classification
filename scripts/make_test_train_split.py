# Alexander Symons | make_test_train_split.py | Aug 24 2023

try:
    import pandas as pd
except:
    print("Required package pandas not available")
    exit()

try:
    import argparse
except:
    print("Required package argparse not available")
    exit()

def load_file(path):
    # Read data
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("Invalid path")
        exit(1)
    return data

# For naming consistency with VALENCIA
def format_VALENCIA(data, read_count_col='total_reads', sample_id_col='Sample_number_for_SRA', label_col="HC_subCST", 
                non_data=['Val_CST', 'Val_subCST','I-A_sim','I-B_sim','II_sim','III-A_sim','III-B_sim', 'IV-A_sim','IV-B_sim',
                'IV-C0_sim', 'IV-C1_sim', 'IV-C2_sim','IV-C3_sim','IV-C4_sim','V_sim', 'Subject_number', "HC_CST"]):

    # We don't need random other data that may be in file, ex: VALECIA's prediction values
    for col in non_data:
        try:
            data = data.drop(columns=col)
        except KeyError:
            print(f"Coun't find non-data column '{col}', continuing")
            continue    # Column wasn't there. Likely preprocessed
    
    # Can't create this row. If it doesn't exist, must be test data
    if label_col == "None":
        print("Unlabeled data")
    elif not label_col in data.keys():
        print(f"HC_subCST equivalent '{label_col}' not found. Please use the column containing target values")
        exit(1)

    # Make sure row exists, if not create it
    if sample_id_col =="None":
        # SampleIDs are [0,n-1]
        data['sampleID'] = range(len(data))
    elif not sample_id_col in data.keys():
        print(data.keys())
        print(f"sampleID equivalent '{sample_id_col}' not found. Use None to fill with range(n)")
        exit(1)

    if read_count_col == "None":
        # Sum all count data columns to get total for each row, and add it as a column
        non_count = [sample_id_col]
        if label_col != "None": non_count += label_col
        total = lambda i:sum(map(lambda x:float(x), list(data.drop(non_count, axis=1).iloc[i])))
        row_totals = [total(row) for row in range(len(data))]
        data['read_count'] = row_totals
    elif not read_count_col in data.keys():
        print(f"read_count equivalent '{read_count_col}' not found. Use None if already normalized")
        exit(1)

    # If corresponding row exists, rename it for valencia/nn_classifier
    data = data.rename(columns={read_count_col:'read_count', sample_id_col:'sampleID', label_col:'HC_subCST'})

    return data

# Split the (formatted) data as requested
def split(data, split, tolerance, labeled):
    # NOTE: HC_subCST may not be an accurate name depending on the application, but it is used because
    # VALENCIA and its formatting were used to standardize testing between it and the 
    # generalized neural classifier.

    # Shuffle data function
    shuffle = lambda x:x.sample(frac=1).reset_index(drop=True)

    # Reorder columns like VALENCIA TODO: reorder subCST
    cols = list(data.keys())

    try:
        cols.remove("sampleID")
        cols.remove("read_count")
    except ValueError:
        print("read_count or sampleID not in data.")
        exit(1)
    
    cols = ["sampleID", "read_count"] + cols

    # Check for duplicates. They don't work in reindex or training
    duplicates = [x for x in cols if cols.count(x) > 1]
    if len(duplicates) != 0:
        print(f"Duplicate values found in data: {' '.join(duplicates)}")
        exit(1)

    data = data.reindex(columns=cols)

    # For unlabeled data no further processing required
    if not labeled:
        train_count = round(split / 100)
        return shuffle(data[:train_count]), shuffle(data[train_count:])

    # Found out what proportion of the data each CST makes. 
    entries = data.groupby(["HC_subCST"]).count()['sampleID']
    total_entries = entries.sum()
    prevelance = entries/total_entries
    train_count_by_category = (entries * (split / 100)).astype(int)

    # Create each set by putting first x values (as defined in train_count_by_category) 
    # in train, and rest in test for each subCST we are controlling for
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()

    subCSTs = list(entries.keys())
    for subCST in subCSTs:
        subCST_entries = data.loc[data["HC_subCST"] == subCST]

        subCST_entries = subCST_entries.sample(frac=1).reset_index(drop=True) # Shuffle and return all data with new indicies

        train_set = pd.concat([train_set, subCST_entries[:train_count_by_category[subCST]]])
        test_set = pd.concat([test_set, subCST_entries[train_count_by_category[subCST]:]])

    # Check data against unsplit data by proportion of each label type
    train_entries = train_set.groupby(['HC_subCST']).count()['sampleID']
    train_total_entries = train_entries.sum()
    train_prevelance = train_entries/train_total_entries

    test_entries = test_set.groupby(['HC_subCST']).count()['sampleID']
    test_total_entries = test_entries.sum()
    test_prevelance = test_entries/test_total_entries

    if ((abs(train_prevelance - prevelance) > tolerance).any() or (abs(test_prevelance - prevelance) > tolerance).any()):
        print("Error: problem when splitting data")
        print(f"Prevalance: {prevelance} \nTrain: {train_prevelance} \nTest: {test_prevelance}")
        exit(2)

    # Shuffle SubCST data
    train_set = shuffle(train_set)
    test_set = shuffle(test_set)

    return train_set, test_set

# Write sets to the path
def write(train_set, test_set, path):
    path = path if path != None else "out"

    train_path = path + "_train.csv"
    test_path = path + "_test.csv"

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)

# Transpose the data preserving columns
def transpose_data(data):
    # Save first column name, and transpose
    first_col = data.keys()[0]
    data = data.T

    # Make new index column and recover first column
    data['index'] = range(len(data))
    data[first_col] = list(data.index)
    data = data.set_index('index')
    
    # Give columns correct labels (names were swapped with index) and get rid of col name column
    data = data.rename(columns={og:new for og,new in zip(list(data.keys()), list(data.iloc[0]))})
    data = data.drop(index=0, axis=0)
    return data

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description="This tool splits VALENCIA data into a testing and training split for training")

    required = parser.add_argument_group("Required arguments")
    required.add_argument("-i", "--input", help="Path to input CSV file",required=True)
    required.add_argument("-o","--output", help="Output file path/prefix")
    required.add_argument("-s","--train-split", default=60, help="Percentage of data for training")
    required.add_argument("-t","--tolerance", default=0.001, help="Decimal for tolerance between orginal data and outputs for checking")
    required.add_argument("-nd","--non-data", default=None, help="Columns which do not contain count data, and shouldn't be passed to classifier")
    required.add_argument("-rc", "--read_counts", default=None, help="Column containing the number of reads for each sample")
    required.add_argument("-sid", "--sample_ids", default=None, help="Column contianing all sample IDs")
    required.add_argument("-lc", "--label_col", default=None, help="Column for community subtypes, renamed to HC_subCST")
    required.add_argument("-tr", "--transpose", default="False", help="Set true if samples are aligned vertically")

    # Read arguments into parser
    args = parser.parse_args()

    # Validate/format arguments
    try:
        train_split = float(args.train_split)
    except TypeError:
        print("Inappropriate type for train split. Must be numerical")
        exit(1)

    try:
        tolerance = float(args.tolerance)
    except TypeError:
        print("Inappropriate type for tolerance. Must be float")
        exit(1)

    transpose = True if args.transpose == "True" else False if args.transpose == "False" else None
    if transpose == None:
        print(f"Transpose should be 'True' or 'False', not '{args.transpose}'")
        exit(1)

    # Default args for nd, rc, sid, lc are all for VALENCIA data set. Different from 'None' argument
    non_data = ['Val_CST', 'Val_subCST','I-A_sim','I-B_sim','II_sim','III-A_sim','III-B_sim', 'IV-A_sim','IV-B_sim',
                'IV-C0_sim', 'IV-C1_sim', 'IV-C2_sim','IV-C3_sim','IV-C4_sim','V_sim', 'Subject_number', "HC_CST"] \
                if args.non_data == None else args.non_data.split(',')
    read_count_col = 'total_reads' if args.read_counts == None else args.read_counts
    sample_id_col = 'Sample_number_for_SRA' if args.sample_ids == None else args.sample_ids
    label_col = "HC_subCST" if args.label_col == None else args.label_col
    labeled = args.label_col != "None"

    # Load, format, split, and save
    data = load_file(args.input)
    if transpose:
        data = transpose_data(data)
    print(data.head())

    # Remove duplicates TODO: Needs work
    print(list(data.keys()).count("Actinobacteria"))
    data = data.loc[:,~data.columns.duplicated()].copy()
    print(list(data.keys()).count("Actinobacteria"))
    
    data = format_VALENCIA(data, read_count_col, sample_id_col, label_col, non_data)
    train_set, test_set = split(data, train_split, tolerance, labeled)
    write(train_set, test_set, args.output)