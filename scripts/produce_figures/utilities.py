import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
import pacmap
import matplotlib as mpl
from matplotlib import cm

# Define function to find accuracy of model
def accuracy_test(lbls, predictions):
    # Do argmax to get index, so as not to do torch.eq on a 2d array
    correct = sum(np.equal(lbls, predictions))
    return (correct / len(predictions)) * 100

# Load classifications from tool along with sample IDs
def load_class(path, class_lbl="subCST", sample_lbl="sampleID"):
    if not path.endswith(".csv"): path += ".csv"
    df = pd.read_csv(path)
    df = df[[sample_lbl, class_lbl]]
    df = df.sort_values(by=sample_lbl)
    df = df.rename(columns={class_lbl: "subCST", sample_lbl: "sampleID"})
    return df

def plot_bars(validation_60, validation_80, valencia_60_data, valencia_80_data, stratabionn_60_data, stratabionn_80_data, forest_60_data, forest_80_data):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    valencia_60_accuracy = accuracy_test(validation_60, valencia_60_data)
    valencia_80_accuracy = accuracy_test(validation_80, valencia_80_data)
    strata_60_accuracy = accuracy_test(validation_60, stratabionn_60_data)
    strata_80_accuracy = accuracy_test(validation_80, stratabionn_80_data)
    forest_60_accuracy = accuracy_test(validation_60, forest_60_data)
    forest_80_accuracy = accuracy_test(validation_80, forest_80_data)

    valencia_60_f1 = f1_score(validation_60, valencia_60_data, average="weighted")
    valencia_80_f1 = f1_score(validation_80, valencia_80_data, average="weighted")
    strata_60_f1 = f1_score(validation_60, stratabionn_60_data, average="weighted")
    strata_80_f1 = f1_score(validation_80, stratabionn_80_data, average="weighted")
    forest_60_f1 = f1_score(validation_60, forest_60_data, average="weighted")
    forest_80_f1 = f1_score(validation_80, forest_80_data, average="weighted")

    valencia_60_precision = precision_score(validation_60, valencia_60_data, average="weighted")
    valencia_80_precision = precision_score(validation_80, valencia_80_data, average="weighted")
    strata_60_precision = precision_score(validation_60, stratabionn_60_data, average="weighted")
    strata_80_precision = precision_score(validation_80, stratabionn_80_data, average="weighted")
    forest_60_precision = precision_score(validation_60, forest_60_data, average="weighted")
    forest_80_precision = precision_score(validation_80, forest_80_data, average="weighted")

    valencia_60_recall = recall_score(validation_60, valencia_60_data, average="weighted")
    valencia_80_recall = recall_score(validation_80, valencia_80_data, average="weighted")
    strata_60_recall = recall_score(validation_60, stratabionn_60_data, average="weighted")
    strata_80_recall = recall_score(validation_80, stratabionn_80_data, average="weighted")
    forest_60_recall = recall_score(validation_60, forest_60_data, average="weighted")
    forest_80_recall = recall_score(validation_80, forest_80_data, average="weighted")

    # axes[0,0].set_title("Accuracy")
    # axes[1,0].set_title("F1 Score")
    # axes[0,1].set_title("Recall")
    # axes[1,1].set_title("Precision")
    axes[0,0].set_title("A)", loc='left')
    axes[1,0].set_title("B)", loc='left')
    axes[0,1].set_title("C)", loc='left')
    axes[1,1].set_title("D)", loc='left')

    colors = ["red", "blue", "lightblue", "green", "lightgreen"]

    # TODO: Should we use these?
    axes[0,0].set_ylim(valencia_60_accuracy*0.9, 100)
    axes[1,0].set_ylim(valencia_60_f1*0.9, 1)
    axes[0,1].set_ylim(valencia_60_recall*0.9, 1)
    axes[1,1].set_ylim(valencia_60_precision*0.9, 1)
    
    bars = [[None for _ in range(2)] for _ in range(2)]

    bars[0][0]=axes[0,0].bar([1,2,3,4,5,6], [valencia_60_accuracy, valencia_80_accuracy, 
                                strata_60_accuracy, strata_80_accuracy, 
                                forest_60_accuracy, forest_80_accuracy],
                                color=colors,
                                tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                                "Stratabion (60% training)", "Stratabion (80% training)", 
                                "Random forest (60% training)", "Random Forest (80% training)"])
    axes[0,0].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabion (60% training)", 
                                "Stratabion (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')

    bars[0][1]=axes[0,1].bar([1,2,3,4,5,6], [valencia_60_f1, valencia_80_f1, 
                            strata_60_f1, strata_80_f1, 
                            forest_60_f1, forest_80_f1],
                            color=colors,
                            tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                            "Stratabion (60% training)", "Stratabion (80% training)", 
                            "Random forest (60% training)", "Random Forest (80% training)"])
    axes[0,1].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabion (60% training)", 
                                "Stratabion (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')

    bars[1][0]=axes[1,0].bar([1,2,3,4,5,6], [valencia_60_recall, valencia_80_recall, 
                            strata_60_recall, strata_80_recall, 
                            forest_60_recall, forest_80_recall],
                            color=colors,
                            tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                            "Stratabion (60% training)", "Stratabion (80% training)", 
                            "Random forest (60% training)", "Random Forest (80% training)"])
    axes[1,0].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabion (60% training)", 
                                "Stratabion (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')

    bars[1][1]=axes[1,1].bar([1,2,3,4,5,6], [valencia_60_precision, valencia_80_precision, 
                            strata_60_precision, strata_80_precision, 
                            forest_60_precision, forest_80_precision],
                            color=colors,
                            tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                            "Stratabion (60% training)", "Stratabion (80% training)", 
                            "Random forest (60% training)", "Random Forest (80% training)"])
    axes[1,1].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabion (60% training)", 
                                "Stratabion (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')


    for x, y in [(0,0), (0,1), (1,0), (1,1)]:
        axes[x,y].set_xlabel("Tool Name")
        axes[x,y].set_ylabel(f"Value {'(%)' if (x,y) == (0,0) else '(decimal)'}")
        axes[x,y].bar_label(bars[x][y])

    plt.tight_layout()
    # plt.title("Performance Metrics for all Methods")
    plt.savefig("fig1.jpeg")
    plt.show()

def plot_confusion(validation_80, valencia_80_data, stratabionn_80_data, forest_80_data):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    all_lbls = sorted(list(set(validation_80)))


    def lbl_to_idx(lbl):
        return all_lbls.index(lbl)
    
    def idx_to_lbl(idx):
        return all_lbls[idx]
    
    validation_80 = list(map(lbl_to_idx, validation_80))
    valencia_80_data = list(map(lbl_to_idx, valencia_80_data))
    stratabionn_80_data = list(map(lbl_to_idx, stratabionn_80_data))
    forest_80_data = list(map(lbl_to_idx, forest_80_data))

    # Valencia, strata, rf, strata - Valencia. All 80% 
    confmat_valencia = confusion_matrix(validation_80, valencia_80_data)
    confmat_stratabionn = confusion_matrix(validation_80, stratabionn_80_data)
    confmat_random_forest = confusion_matrix(validation_80, forest_80_data)
    # cm_val_strata_diff = confusion_matrix(reference_data, valencia_data)

    # blues_cm = mpl.colormaps.get_cmap('Blues')
    # reds_cm = mpl.colormaps.get_cmap('Reds')
    # colors = blues_cm(np.linspace(0, 1, 128))
    # colors[:64, :] = reds_cm(np.linspace(1, 0, 64))
    # colors[64:, :] = blues_cm(np.linspace(0, 1, 64))
    # print(colors[0])
    # RedBlues = ListedColormap(colors)

    from matplotlib.colors import ListedColormap
    blues_cm = mpl.colormaps.get_cmap("RdBu")
    colors = blues_cm(np.linspace(0, 1, 128))
    # colors[:24, 3] = 0.7
    # colors[-24:, 3] = 0.7
    colors[:, 3] = 0.65
    RedBlues = ListedColormap(colors)

    cms = [confmat_valencia, confmat_stratabionn, confmat_random_forest, confmat_stratabionn - confmat_valencia]
    cmaps = ["Blues", "Blues", "Blues", RedBlues]
    coords = [(0,0), (0,1), (1,0), (1,1)]
    # tools = ["VALENCIA", "Stratabionn", "Random Forest", "Stratabionn - VALENCIA"]
    tools = ["A)", "B)", "C)", "D)"]

    for i in range(len(cms)):
        x, y = coords[i]
        disp = ConfusionMatrixDisplay(confusion_matrix=cms[i], display_labels=all_lbls)
        disp.plot(ax=axes[x, y], cmap=cmaps[i])
        axes[x,y].set_xticklabels(all_lbls, rotation=45)
        axes[x,y].set_title(tools[i], loc='left')
    
    plt.tight_layout()
    # plt.title("Confusion Matrices for all Methods")
    plt.savefig("fig2.jpeg")
    plt.show()

def extract_numpy(df: pd.DataFrame, label_col:str, drop_cols:list[str] = [], norm:bool = True) -> (np.array, np.array, list):
    #normalized_data = df.drop(columns=["sampleID", "HC_subCST"]).astype(float)

    if label_col != None:
        required_nondata = [label_col] + drop_cols

        labels = list(df[label_col])
        all_labels = sorted(list(set(labels)))
        y_labels = np.array(list(map(lambda x:all_labels.index(x), labels)))
    else:
        required_nondata = drop_cols

        y_labels = np.array([0] * len(df))
        all_labels = ["Unlabeled"]

    for c in required_nondata:
        if c not in list(df.columns):
            print("ERR:", c, "missing from data")
            exit(1)

    normalized_data = df.drop(columns=required_nondata).astype(float)

    if norm:
        normalized_data = normalized_data.div(normalized_data.sum(axis=1), axis=0)
        normalized_data[normalized_data.isnull()] = 1.0e-5
        normalized_data[normalized_data.eq(0)] = 1.0e-5

    return normalized_data.to_numpy(), y_labels, all_labels

def plot_single_pacmap(X_data, y_data, universe, ax):   
    # Generate PACMAPs for all data sets.
    cmap = plt.get_cmap("Spectral", len(universe))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(universe)-1)

    embedding = pacmap.PaCMAP()
    X_transformed = embedding.fit_transform(X_data, init="pca")

    s=ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y_data, s=2.6)
    
    elms=s.legend_elements()

    elms = (elms[0],universe)
    ax.legend(*elms, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')

    # handles = [mpl.patches.Patch(color=cmap(norm(i)), label=universe[i]) for i in range(len(universe))]
    # ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    
    # plt.show()
    # plt.savefig(path + "pacmap_test.png", bbox_inches='tight')

def plot_pacmaps(ravel_data, hickey_data, ashley_data, edlund_data):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    dfs = [ravel_data, hickey_data, ashley_data, edlund_data]
    label_col = ["HC_subCST", None, None, "HC_subCST"]
    nondata_cols = [["sampleID", "read_count"], ["sampleID"], ["sampleID", "read_count"], ["sampleID"]]
    coords = [(0,0), (0,1), (1,1), (1,0)]
    # dataset_names = ["Ravel", "Hickey", "Ashley", "Edlund"]
    dataset_names = ["A)", "B)", "C)", "D)"]

    for i in range(len(dfs)):
        x, y = coords[i]
        df = dfs[i]
        lbl_col = label_col[i]
        nd_col = nondata_cols[i]
        print("Graphing", dataset_names[i])

        X_data, y_data, universe = extract_numpy(df, lbl_col, nd_col)
        plot_single_pacmap(X_data, y_data, universe, axes[x, y])

        axes[x, y].set_title(dataset_names[i], loc='left')
    
    plt.tight_layout()
    # plt.title("Confusion Matrices for all Methods")
    plt.savefig("fig3.jpeg")
    plt.show()

def fig_1_2_main(args):
    validation_60_path = args.validation_60
    validation_80_path = args.validation_80
    valencia_60_path = args.valencia_class_60
    valencia_80_path = args.valencia_class_80
    stratabionn_60_path = args.stratabionn_class_60
    stratabionn_80_path = args.stratabionn_class_80
    forest_60_path = args.forest_class_60
    forest_80_path = args.forest_class_80

    # Only validate on validation set. Using everything skews results
    validation_60 = list(load_class(validation_60_path, class_lbl="HC_subCST")["subCST"])
    validation_80 = list(load_class(validation_80_path, class_lbl="HC_subCST")["subCST"])

    valencia_60_data = list(load_class(valencia_60_path)["subCST"])
    valencia_80_data = list(load_class(valencia_80_path)["subCST"])
    stratabionn_60_data = list(load_class(stratabionn_60_path)["subCST"])
    stratabionn_80_data = list(load_class(stratabionn_80_path)["subCST"])
    forest_60_data = list(load_class(forest_60_path)["subCST"])
    forest_80_data = list(load_class(forest_80_path)["subCST"])

    plot_bars(validation_60, validation_80, valencia_60_data, valencia_80_data, stratabionn_60_data, stratabionn_80_data, forest_60_data, forest_80_data)

    plot_confusion(validation_80, valencia_80_data, stratabionn_80_data, forest_80_data)


def fig_3_main(args):
    ravel_path = args.ravel_data
    hickey_path = args.hickey_data
    ashley_path = args.ashley_data
    edlund_path = args.edlund_data

    ravel_data = pd.read_csv(ravel_path)
    hickey_data = pd.read_csv(hickey_path)
    ashley_data = pd.read_csv(ashley_path)
    edlund_data = pd.read_csv(edlund_path)

    plot_pacmaps(ravel_data, hickey_data, ashley_data, edlund_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots various graphs")

    subparsers = parser.add_subparsers(dest='subcommand', help='Available subcommands')

    parser_fig_1_2 = subparsers.add_parser('fig_1_and_2', help='Figure 1 and 2 help')
    parser_fig_3 = subparsers.add_parser('fig_3', help='Figure 3 help')

    # Argument group for figure 1
    fig_1_2_group = parser_fig_1_2.add_argument_group('Fig 1 and 2 Options')
    fig_1_2_group.add_argument("--valencia-class-60", help="Classifications from VALENCIA", type=str)
    fig_1_2_group.add_argument("--valencia-class-80", help="Classifications from VALENCIA", type=str)
    fig_1_2_group.add_argument("--stratabionn-class-60", help="Classifications from Stratabionn 60/20/20", type=str)
    fig_1_2_group.add_argument("--stratabionn-class-80", help="Classifications from Stratabionn 80/10/10", type=str)
    fig_1_2_group.add_argument("--forest-class-60", help="Classifications from Random Forest 60/20/20", type=str)
    fig_1_2_group.add_argument("--forest-class-80", help="Classifications from Random Forest 80/10/10", type=str)
    fig_1_2_group.add_argument("--validation-60", help="Classifications from 60% validaiton set", type=str)
    fig_1_2_group.add_argument("--validation-80", help="Classifications from 80% validaiton set", type=str)

    # Argument group for figure 1
    fig_3_group = parser_fig_3.add_argument_group('Fig 3 Options')
    fig_3_group.add_argument("--ravel-data", help="Path to Ravel et al. data (VALENCIA formatted)", type=str)
    fig_3_group.add_argument("--hickey-data", help="Path to Hickey et al. data (VALENCIA formatted)", type=str)
    fig_3_group.add_argument("--ashley-data", help="Path to Ashley et al. data (VALENCIA formatted)", type=str)
    fig_3_group.add_argument("--edlund-data", help="Path to Edlund et al. data (VALENCIA formatted)", type=str)


    # Parse arguments
    args = parser.parse_args()

    if args.subcommand == 'fig_1_and_2':
        fig_1_2_main(args)
    elif args.subcommand == 'fig_3':
        fig_3_main(args)
    else:
        parser.print_help()