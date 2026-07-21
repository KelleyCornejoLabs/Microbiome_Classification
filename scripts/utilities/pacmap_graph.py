import pacmap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

import argparse

def extract_numpy(df: pd.DataFrame, norm:bool = True, label:str = "HC_subCST", non_data_columns:list = ["sampleID","read_count"], custom_order = None):
    #normalized_data = df.drop(columns=["sampleID", "HC_subCST"]).astype(float)
    normalized_data = df.drop(columns=non_data_columns + [label]).astype(float)

    if custom_order is not None:
        df[label] = pd.Categorical(df[label], categories=custom_order, ordered=True)
        df = df.sort_values(label)

    labels = list(df[label])
    all_labels = list(set(labels))
    print("INFO: All labels:", ",".join(list(map(str, all_labels))))
    y_labels = np.array(list(map(lambda x:all_labels.index(x), labels)))

    if norm:
        normalized_data = normalized_data.div(normalized_data.sum(axis=1), axis=0)
        normalized_data[normalized_data.isnull()] = 1.0e-5
        normalized_data[normalized_data.eq(0)] = 1.0e-5

    return normalized_data.to_numpy(), y_labels, all_labels

def generate_pacmap_graph(X_data, y_data, universe, out_path):
    embedding = pacmap.PaCMAP()
    X_transformed = embedding.fit_transform(X_data, init="pca")

    cmap = plt.get_cmap("Spectral", len(universe))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(universe)-1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    s=ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y_data, s=2.6)
    # elms=s.legend_elements()

    # elms = (elms[0],universe)
    # print(elms)
    # print(len(set(y_data)), len(elms[0]), len(elms[1]))
    # ax.legend(*elms)

    handles = [mpl.patches.Patch(color=cmap(norm(i)), label=universe[i]) for i in range(len(universe))]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    
    # plt.show()
    plt.tight_layout()
    plt.savefig(out_path + ".jpeg")
    plt.savefig(out_path + ".svg", format='svg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make pacmap graph of microbiome data.")
    required = parser.add_argument_group("Required arguments")
    required.add_argument("-i", "--input", type=str, required=True, help="Valencia formatted data")
    required.add_argument("-l", "--label", default="HC_subCST", required=False, type=str, help="Labels for the data, e.g. HC_subCST")
    required.add_argument("-o", "--output", type=str, default="pacmap_gut_labels.jpeg", required=False, help="Out path for the graph")
    required.add_argument("-nd", "--non-data", default="sampleID,read_count", required=False, type=str, help="Comma seperated list of non-data column names")
    args = parser.parse_args()

    in_path = args.input
    label = args.label
    out_path = args.output
    non_data_columns = args.non_data.split(",")

    #df = pd.read_csv("../oral_test_set/processed_data.csv")
    # df = pd.read_csv("../ref_set/all_samples_taxonomic_composition_data.csv")
    # df = pd.read_csv("../gut/training_Set/gut_no_location_is_control_80_10_10_train.csv")
    # df = pd.read_csv("../gut/training_Set/gut_no_location_specific_label_80_10_10_test.csv")
    print("INFO: Reading data")
    df = pd.read_csv(in_path)

    print("INFO: Extracting data")
    X_data, y_data, universe = extract_numpy(df, label=label, non_data_columns=non_data_columns, custom_order = ["oceania", "south america", "africa", "europe", "north america", "asia"][::-1])
    print("INFO: All read labels:", ",".join(universe))
    # exit()
    
    print("INFO: Running pacmap")
    generate_pacmap_graph(X_data, y_data, universe, out_path)
