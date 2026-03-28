import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
import pacmap
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

blair_colors = ["#E73F74","#F1CE63","#77AADD","#009988","#9467BD","#FF9D9A","#99DDFF","#AAAA00","#225522","#882255","#997700","#4B4B8F","#8C564B"]
blair_cmap = LinearSegmentedColormap.from_list("custom", blair_colors)#plt.get_cmap("gist_rainbow")

# Normalize the data so its all the same
def str_norm(x: str):
    prefixes = ["g_", "o_", "k_", "f_", "d_", "c_"]

    x = x.lower().replace(' ', '_')

    # Remove any prefix
    if x[1] == '_':
        start = 3 if x[2] == '_' else 2
        x = x[start:]

    # Return lower_camel_case
    return x

def normalize(x):
    return x.div(x.sum(axis=1), axis=0)

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

def plot_bars_oral(validation_60, validation_80, stratabionn_60_data, stratabionn_80_data):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    print(len(validation_60), len(stratabionn_60_data))
    print(len(validation_80), len(stratabionn_80_data))

    strata_60_accuracy = round(accuracy_test(validation_60, stratabionn_60_data) / 100, 3)
    strata_80_accuracy = round(accuracy_test(validation_80, stratabionn_80_data) / 100, 3)

    strata_60_f1 = round(f1_score(validation_60, stratabionn_60_data, average="weighted"), 3)
    strata_80_f1 = round(f1_score(validation_80, stratabionn_80_data, average="weighted"), 3)

    strata_60_precision = round(precision_score(validation_60, stratabionn_60_data, average="weighted"), 3)
    strata_80_precision = round(precision_score(validation_80, stratabionn_80_data, average="weighted"), 3)

    strata_60_recall = round(recall_score(validation_60, stratabionn_60_data, average="weighted"), 3)
    strata_80_recall = round(recall_score(validation_80, stratabionn_80_data, average="weighted"), 3)

    # axes[0,0].set_title("Accuracy")
    # axes[0,1].set_title("Recall")
    # axes[1,0].set_title("F1 Score")
    # axes[1,1].set_title("Precision")
    # ax.set_title("A)", loc='left')

    colors = ["red", "lightcoral", "blue", "lightblue", "green", "lightgreen", "orchid", "plum"]

    # TODO: Should we use these?
    ax.set_ylim(0.9, 1)
    

    bars=ax.bar([1,2,3,4,5,6,7,8], [strata_60_accuracy, strata_80_accuracy, 
                                strata_60_f1, strata_80_f1, 
                                strata_60_precision, strata_80_precision,
                                strata_60_recall, strata_80_recall],
                                color=colors,
                                tick_label=["Accuracy (60% training)", "Accuracy (80% training)", 
                                "F1 (60% training)", "F1 (80% training)", 
                                "Precision (60% training)", "Precision (80% training)",
                                "Recall (60% training)", "Recall (80% training)"])
    ax.set_xticklabels(["Accuracy (60% training)", "Accuracy (80% training)", 
                                "F1 (60% training)", "F1 (80% training)", 
                                "Precision (60% training)", "Precision (80% training)",
                                "Recall (60% training)", "Recall (80% training)"], rotation=45, ha='right')



    # for x,y in [(0,0), (0,1), (1,0), (1,1)]:
    #     for c in axes[x,y].containers: axes[x,y].bar_label(c, fmt='%2.3f', label_type='center')

    ax.set_xlabel("Metric")
    ax.set_ylabel(f"Value (decimal)")
    ax.bar_label(bars)

    plt.tight_layout()
    # plt.title("Performance Metrics for all Methods")
    plt.savefig("fig6.jpeg")
    plt.savefig("fig6.svg", format='svg')
    plt.show()

def plot_bars(validation_60, validation_80, valencia_60_data, valencia_80_data, stratabionn_60_data, stratabionn_80_data, forest_60_data, forest_80_data):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    print(len(validation_60), len(valencia_60_data))
    print(len(validation_80), len(valencia_80_data))
    print(len(validation_60), len(stratabionn_60_data))
    print(len(validation_80), len(stratabionn_80_data))
    print(len(validation_60), len(forest_60_data))
    print(len(validation_80), len(forest_80_data))

    valencia_60_accuracy = round(accuracy_test(validation_60, valencia_60_data) / 100, 3)
    valencia_80_accuracy = round(accuracy_test(validation_80, valencia_80_data) / 100, 3)
    strata_60_accuracy = round(accuracy_test(validation_60, stratabionn_60_data) / 100, 3)
    strata_80_accuracy = round(accuracy_test(validation_80, stratabionn_80_data) / 100, 3)
    forest_60_accuracy = round(accuracy_test(validation_60, forest_60_data) / 100, 3)
    forest_80_accuracy = round(accuracy_test(validation_80, forest_80_data) / 100, 3)

    valencia_60_f1 = round(f1_score(validation_60, valencia_60_data, average="weighted"), 3)
    valencia_80_f1 = round(f1_score(validation_80, valencia_80_data, average="weighted"), 3)
    strata_60_f1 = round(f1_score(validation_60, stratabionn_60_data, average="weighted"), 3)
    strata_80_f1 = round(f1_score(validation_80, stratabionn_80_data, average="weighted"), 3)
    forest_60_f1 = round(f1_score(validation_60, forest_60_data, average="weighted"), 3)
    forest_80_f1 = round(f1_score(validation_80, forest_80_data, average="weighted"), 3)

    print(len(set(validation_60)), len(set(valencia_60_data)), len(set(stratabionn_60_data)))
    valencia_60_precision = round(precision_score(validation_60, valencia_60_data, average="weighted"), 3)
    valencia_80_precision = round(precision_score(validation_80, valencia_80_data, average="weighted"), 3)
    strata_60_precision = round(precision_score(validation_60, stratabionn_60_data, average="weighted"), 3)
    strata_80_precision = round(precision_score(validation_80, stratabionn_80_data, average="weighted"), 3)
    forest_60_precision = round(precision_score(validation_60, forest_60_data, average="weighted"), 3)
    forest_80_precision = round(precision_score(validation_80, forest_80_data, average="weighted"), 3)

    valencia_60_recall = round(recall_score(validation_60, valencia_60_data, average="weighted"), 3)
    valencia_80_recall = round(recall_score(validation_80, valencia_80_data, average="weighted"), 3)
    strata_60_recall = round(recall_score(validation_60, stratabionn_60_data, average="weighted"), 3)
    strata_80_recall = round(recall_score(validation_80, stratabionn_80_data, average="weighted"), 3)
    forest_60_recall = round(recall_score(validation_60, forest_60_data, average="weighted"), 3)
    forest_80_recall = round(recall_score(validation_80, forest_80_data, average="weighted"), 3)

    # axes[0,0].set_title("Accuracy")
    # axes[1,0].set_title("F1 Score")
    # axes[0,1].set_title("Recall")
    # axes[1,1].set_title("Precision")
    axes[0,0].set_title("A)", loc='left')
    axes[1,0].set_title("B)", loc='left')
    axes[0,1].set_title("C)", loc='left')
    axes[1,1].set_title("D)", loc='left')

    colors = ["red", "lightcoral", "blue", "lightblue", "green", "lightgreen"]

    # TODO: Should we use these?
    axes[0,0].set_ylim(valencia_60_accuracy*0.9, 1)
    axes[0,1].set_ylim(valencia_60_f1*0.9, 1)
    axes[1,0].set_ylim(valencia_60_recall*0.9, 1)
    axes[1,1].set_ylim(valencia_60_precision*0.9, 1)
    
    
    bars = [[None for _ in range(2)] for _ in range(2)]

    bars[0][0]=axes[0,0].bar([1,2,3,4,5,6], [valencia_60_accuracy, valencia_80_accuracy, 
                                strata_60_accuracy, strata_80_accuracy, 
                                forest_60_accuracy, forest_80_accuracy],
                                color=colors,
                                tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                                "Stratabionn (60% training)", "Stratabionn (80% training)", 
                                "Random forest (60% training)", "Random Forest (80% training)"])
    axes[0,0].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabionn (60% training)", 
                                "Stratabionn (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')

    bars[0][1]=axes[0,1].bar([1,2,3,4,5,6], [valencia_60_f1, valencia_80_f1, 
                            strata_60_f1, strata_80_f1, 
                            forest_60_f1, forest_80_f1],
                            color=colors,
                            tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                            "Stratabionn (60% training)", "Stratabionn (80% training)", 
                            "Random forest (60% training)", "Random Forest (80% training)"])
    axes[0,1].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabionn (60% training)", 
                                "Stratabionn (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')

    bars[1][0]=axes[1,0].bar([1,2,3,4,5,6], [valencia_60_recall, valencia_80_recall, 
                            strata_60_recall, strata_80_recall, 
                            forest_60_recall, forest_80_recall],
                            color=colors,
                            tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                            "Stratabionn (60% training)", "Stratabionn (80% training)", 
                            "Random forest (60% training)", "Random Forest (80% training)"])
    axes[1,0].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabionn (60% training)", 
                                "Stratabionn (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')

    bars[1][1]=axes[1,1].bar([1,2,3,4,5,6], [valencia_60_precision, valencia_80_precision, 
                            strata_60_precision, strata_80_precision, 
                            forest_60_precision, forest_80_precision],
                            color=colors,
                            data=[1,2,3,4,5,6],
                            tick_label=["Valencia (60% training)", "Valencia (80% training)", 
                            "Stratabionn (60% training)", "Stratabionn (80% training)", 
                            "Random forest (60% training)", "Random Forest (80% training)"])
    axes[1,1].set_xticklabels(["Valencia (60% training)", "Valencia (80% training)", "Stratabionn (60% training)", 
                                "Stratabionn (80% training)", "Random forest (60% training)", 
                                "Random Forest (80% training)"], rotation=45, ha='right')



    # for x,y in [(0,0), (0,1), (1,0), (1,1)]:
    #     for c in axes[x,y].containers: axes[x,y].bar_label(c, fmt='%2.3f', label_type='center')

    for x, y in [(0,0), (0,1), (1,0), (1,1)]:
        axes[x,y].set_xlabel("Tool Name")
        axes[x,y].set_ylabel(f"Value {'(%)' if (x,y) == (0,0) else '(decimal)'}")
        axes[x,y].bar_label(bars[x][y])

    plt.tight_layout()
    # plt.title("Performance Metrics for all Methods")
    plt.savefig("fig1.svg", format='svg')
    plt.savefig("fig1.jpeg")
    plt.savefig("fig1.pdf", format="pdf")
    plt.show()

def plot_confusion_oral(validation_80, stratabionn_80_data):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    all_lbls = sorted(list(set(validation_80)))


    def lbl_to_idx(lbl):
        return all_lbls.index(lbl)
    
    def idx_to_lbl(idx):
        return all_lbls[idx]
    
    validation_80 = list(map(lbl_to_idx, validation_80))
    stratabionn_80_data = list(map(lbl_to_idx, stratabionn_80_data))

    confmat_stratabionn = confusion_matrix(validation_80, stratabionn_80_data)

    disp = ConfusionMatrixDisplay(confusion_matrix=confmat_stratabionn, display_labels=all_lbls)
    disp.plot(ax=ax, cmap="Blues")
    ax.set_xticklabels(all_lbls, rotation=45)
    ax.set_title("Oral Dataset: 80% Training", loc='left')
    
    plt.tight_layout()
    # plt.title("Confusion Matrices for all Methods")
    plt.savefig("fig7.svg", format='svg')
    plt.savefig("fig7.jpeg")
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
    plt.savefig("fig2.svg", format='svg')
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

def plot_single_pacmap(X_data, y_data, universe, ax, divisions=[-1], marks=['o']):   
    # Generate PACMAPs for all data sets.
    embedding = pacmap.PaCMAP()
    X_transformed = embedding.fit_transform(X_data, init="pca")

    # cmap = plt.get_cmap("gist_rainbow")
    cmap = blair_cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=len(universe)-1)
    def apply_col(x):
        col = list(cmap(norm(x)))
        if "ref" in universe[x]:
            col = (np.float64(col[0]),np.float64(col[1]),np.float64(col[2]), np.float64(0.025))
        return col

    adjusted_cmap = ListedColormap([apply_col(i) for i in range(0, len(universe))])

    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap=adjusted_cmap, marker="x", c=y_data, s=2.6)
    # start = 0
    # for i, division in enumerate(divisions):
    #     end = division
    #     print(marks[i])
    #     ax.scatter(X_transformed[start:end, 0], X_transformed[start:end, 1], cmap=apply_col, marker=marks[i], c=y_data[start:end], s=2.6)
    #     start = division
    
    # elms=s.legend_elements()

    # elms = (elms[0],universe)
    # print(elms, len(set(y_data)))
    # ax.legend(*elms, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    handles = [mpl.patches.Patch(color=apply_col(i), label=universe[i]) for i in range(len(universe))]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    
    # plt.show()
    # plt.savefig(path + "pacmap_test.png", bbox_inches='tight')

def fig_3(train_data, test_data, validate_data, figure=3):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    # X_data, y_data, universe = extract_numpy(france_data, "HC_subCST", ["sampleID", "read_count"])
    # plot_single_pacmap(X_data, y_data, universe, axes)


    train_class = train_data["HC_subCST"]+" train"
    test_class = test_data["HC_subCST"]+" test"
    validate_class = validate_data["HC_subCST"]+" validate"

    all_data = pd.concat([train_data, test_data, validate_data], axis=0)
    all_class = pd.concat([train_class, test_class, validate_class], axis=0)
    all_data["label"] = all_class
    all_data.to_csv("test.csv")

    X_data, y_data, universe = extract_numpy(all_data, "label", ["HC_subCST", "sampleID", "read_count"])

    embedding = pacmap.PaCMAP()
    X_transformed = embedding.fit_transform(X_data, init="pca")

    train_transformed = []
    test_transformed = []
    validation_transformed = []

    all_labels = sorted(list(set(list(map(lambda x:x.split(" ")[0], universe)))))

    # cmap = sns.color_palette("husl", as_cmap=True)
    cmap = blair_cmap#plt.get_cmap("gist_rainbow")
    norm = mpl.colors.Normalize(vmin=0, vmax=len(all_labels)-1)
    def apply_col(x, opacity=0.05):
        col = list(cmap(norm(x)))
        # if "train" in universe[x]:
        #     col = (np.float64(col[0]),np.float64(col[1]),np.float64(col[2]), np.float64(0.1))
        # elif "test" in universe[x]:
        #     col = (np.float64(col[0]),np.float64(col[1]),np.float64(col[2]), np.float64(0.3))
        return col

    adjusted_cmap = ListedColormap([apply_col(i) for i in range(0, len(all_labels))])
    print(all_labels)


    for i in range(len(y_data)):
        label = universe[y_data[i]].split(" ")[0]
        if "train" in universe[y_data[i]]:
            train_transformed.append((X_transformed[i, 0], X_transformed[i, 1], all_labels.index(label)))
        elif "test" in universe[y_data[i]]:
            test_transformed.append((X_transformed[i, 0], X_transformed[i, 1], all_labels.index(label)))
        elif "validate" in universe[y_data[i]]:
            validation_transformed.append((X_transformed[i, 0], X_transformed[i, 1], all_labels.index(label)))
        else:
            print("ERR: Unknown:", y_data[i])
            exit(1)

    f0 = lambda x : list(map(lambda a:a[0], x))
    f1 = lambda x : list(map(lambda a:a[1], x))
    f2 = lambda x : list(map(lambda a:a[2], x))

    # axes.scatter(X_transformed[:,0], X_transformed[:,1], cmap="gist_rainbow", marker="_", c=y_data, s=2.6)
    axes[0].scatter(f0(train_transformed), f1(train_transformed), cmap=adjusted_cmap, marker="x", c=f2(train_transformed), s=2.6)
    axes[1].scatter(f0(test_transformed), f1(test_transformed), cmap=adjusted_cmap, marker="x", c=f2(test_transformed), s=2.6)
    axes[2].scatter(f0(validation_transformed), f1(validation_transformed), cmap=adjusted_cmap, marker="x", c=f2(validation_transformed), s=2.6)
    # plot_single_pacmap(X_data, y_data, universe, axes, divisions=[len(france_common_data), -1], marks=['+', 'x'])

    handles = [mpl.patches.Patch(color=apply_col(i), label=all_labels[i]) for i in range(0,len(all_labels))]
    axes[0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    axes[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    axes[2].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    axes[0].set_title("A)", loc='left')
    axes[1].set_title("B)", loc='left')
    axes[2].set_title("C)", loc='left')
    
    plt.tight_layout()
    # plt.title("Confusion Matrices for all Methods")
    plt.savefig(f"fig{figure}.svg", format='svg')
    plt.savefig(f"fig{figure}.jpeg")
    plt.show()

def plot_pacmaps(france_data, hickey_data, ashley_data, baker_data, figure=3):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    dfs = [france_data]#, hickey_data]
    label_col = ["HC_subCST"]#, None if figure == 3 else "subCST"]
    nondata_cols = [["sampleID", "read_count"]]#, ["sampleID"]]

    if figure == 6:
        dfs = [baker_data, ashley_data]
        label_col = ["HC_subCST", None]
        nondata_cols = [["sampleID"], ["sampleID", "read_count"]]
    
    dataset_names = ["A)", "B)"]
    for i in range(len(dfs)):
        df = dfs[i]
        lbl_col = label_col[i]
        nd_col = nondata_cols[i]
        print("Graphing", dataset_names[i])

        X_data, y_data, universe = extract_numpy(df, lbl_col, nd_col)
        plot_single_pacmap(X_data, y_data, universe, axes[i])

        axes[i].set_title(dataset_names[i], loc='left')
    
    plt.tight_layout()
    # plt.title("Confusion Matrices for all Methods")
    plt.savefig(f"fig{figure}.svg", format='svg')
    plt.savefig(f"fig{figure}.jpeg")
    plt.show()

# Extract specific columns. Returns normalized column names in order specified
def extract_cols(data: pd.DataFrame, keep_columns):
    data = data.rename(columns=str_norm)
    data = data[keep_columns]
    return data

def plot_hickey_valencia_comparison(france_data, hickey_data, common_cols, figure=4):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    france_common_data = extract_cols(france_data, common_cols)
    hickey_common_data = extract_cols(hickey_data, common_cols)

    france_classification = france_data["HC_subCST"]+" ref"
    hickey_classification = hickey_data["subCST"]

    all_data = pd.concat([france_common_data, hickey_common_data], axis=0)
    all_class = pd.concat([france_classification, hickey_classification], axis=0)
    all_data["label"] = all_class
    all_data.to_csv("test.csv")

    print("Div:", len(france_common_data))

    # universe = list(set(all_class))

    # all_data[all_data.isna()] = 0

    X_data, y_data, universe = extract_numpy(all_data, "label")
    print(list(set(y_data)))

    embedding = pacmap.PaCMAP()
    X_transformed = embedding.fit_transform(X_data, init="pca")

    cmap = blair_cmap#plt.get_cmap("gist_rainbow")
    # cmap = plt.get_cmap("gist_rainbow")

    unique_labels = list(map(lambda x:x.split(" ")[0], universe))

    norm = mpl.colors.Normalize(vmin=0, vmax=len(unique_labels)-1)
    def apply_col(x, opacity=0.05):
        label = universe[x]
        col = list(cmap(norm(x))) #unique_labels.index(label.split(" ")[0])
        if "ref" in universe[x]:
            col = (np.float64(col[0]),np.float64(col[1]),np.float64(col[2]), np.float64(opacity))
        return col

    opaque_cmap = ListedColormap([apply_col(i,1) for i in range(0, len(universe))])
    adjusted_cmap = ListedColormap([apply_col(i) for i in range(0, len(universe))])

    axes[0].set_title("A)", loc='left')
    axes[1].set_title("B)", loc='left')
    axes[0].scatter(X_transformed[:, 0], X_transformed[:, 1], cmap=opaque_cmap, marker="x", c=y_data, s=2.6)
    axes[1].scatter(X_transformed[:, 0], X_transformed[:, 1], cmap=adjusted_cmap, marker="x", c=y_data, s=2.6)
    # plot_single_pacmap(X_data, y_data, universe, axes, divisions=[len(france_common_data), -1], marks=['+', 'x'])

    handles = [mpl.patches.Patch(color=apply_col(i, 1), label=universe[i]) for i in range(len(universe))]
    adj_handles = [mpl.patches.Patch(color=apply_col(i), label=universe[i]) for i in range(len(universe))]
    axes[0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    axes[1].legend(handles=adj_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    
    plt.tight_layout()
    # plt.title("Confusion Matrices for all Methods")
    print(f"Saving to fig{figure}")
    plt.savefig(f"fig{figure}.svg", format='svg')
    plt.savefig(f"fig{figure}.jpeg")
    plt.show()

def plot_3_study_classifications(baseline_data, study_1, study_2, common_cols, figure, study_1_name="Hyuhn", study_2_name="Baker"):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    base_common_data = extract_cols(baseline_data, common_cols)
    study_1_common_data = extract_cols(study_1, common_cols)
    study_2_common_data = extract_cols(study_2, common_cols)

    base_classification = baseline_data["HC_subCST"]+" ref"
    study_1_classification = study_1["subCST"]+" "+study_1_name
    study_2_classification = study_2["subCST"]+" "+study_2_name

    all_data = pd.concat([base_common_data, study_1_common_data, study_2_common_data], axis=0)
    all_class = pd.concat([base_classification, study_1_classification, study_2_classification], axis=0)
    all_data["label"] = all_class
    # all_data.to_csv("test.csv")

    # print("Div:", len(base_common_data))

    # universe = list(set(all_class))

    # all_data[all_data.isna()] = 0

    X_data, y_data, universe = extract_numpy(all_data, "label")
    print(list(set(y_data)))

    embedding = pacmap.PaCMAP()
    X_transformed = embedding.fit_transform(X_data, init="pca")

    cmap = blair_cmap#plt.get_cmap("gist_rainbow")
    # cmap = plt.get_cmap("gist_rainbow")

    unique_labels = list(map(lambda x:x.split(" ")[0], universe))

    print(universe)

    norm = mpl.colors.Normalize(vmin=0, vmax=len(universe))
    def apply_col(x, opacity=0.05, make_opaque=None):
        label = universe[x]
        col = list(cmap(norm(x))) #unique_labels.index(label.split(" ")[0])
        if make_opaque != None and make_opaque not in universe[x]:
            # If make_opaque substring not found in this label, make it transparent
            col = (np.float64(col[0]),np.float64(col[1]),np.float64(col[2]), np.float64(opacity))
        return col

    opaque_cmap = ListedColormap([apply_col(i, 1) for i in range(0, len(universe))])
    show_1_cmap = ListedColormap([apply_col(i, 0.05, study_1_name) for i in range(0, len(universe))])
    show_2_cmap = ListedColormap([apply_col(i, 0.05, study_2_name) for i in range(0, len(universe))])

    axes[0].set_title("A)", loc='left')
    axes[1].set_title("B)", loc='left')
    axes[2].set_title("C)", loc='left')
    axes[0].scatter(X_transformed[:, 0], X_transformed[:, 1], cmap=opaque_cmap, marker="x", c=y_data, s=2.6)
    axes[1].scatter(X_transformed[:, 0], X_transformed[:, 1], cmap=show_1_cmap, marker="x", c=y_data, s=2.6)
    axes[2].scatter(X_transformed[:, 0], X_transformed[:, 1], cmap=show_2_cmap, marker="x", c=y_data, s=2.6)
    # plot_single_pacmap(X_data, y_data, universe, axes, divisions=[len(france_common_data), -1], marks=['+', 'x'])

    handles = [mpl.patches.Patch(color=apply_col(i, 1), label=universe[i]) for i in range(len(universe))]
    adj_1_handles = [mpl.patches.Patch(color=apply_col(i, 0.05, study_1_name), label=universe[i]) for i in range(len(universe))]
    adj_2_handles = [mpl.patches.Patch(color=apply_col(i, 0.05, study_2_name), label=universe[i]) for i in range(len(universe))]
    axes[0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    axes[1].legend(handles=adj_1_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    axes[2].legend(handles=adj_2_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')
    
    plt.tight_layout()
    # plt.title("Confusion Matrices for all Methods")
    print(f"Saving to fig{figure}")
    plt.savefig(f"fig{figure}.svg", format='svg')
    plt.savefig(f"fig{figure}.jpeg")
    plt.show()

def get_pacmap_csv(france_data, hickey_data, common_cols):
    france_common_data = extract_cols(france_data, common_cols)
    hickey_common_data = extract_cols(hickey_data, common_cols)

    france_classification = pd.DataFrame({
        "class": france_data["HC_subCST"],
        "dataset": "France"
    })
    hickey_classification = pd.DataFrame({
        "class": hickey_data["subCST"],
        "dataset": "Hickey"
    })

    all_data = pd.concat([france_common_data, hickey_common_data], axis=0)
    all_class = pd.concat([france_classification, hickey_classification], axis=0)
    all_data = pd.concat([all_class, all_data], axis=1)

    embedding = pacmap.PaCMAP()
    X_transformed = embedding.fit_transform(all_data.drop(columns=["class","dataset"]), init="pca")

    info = pd.DataFrame()
    info["x"] = X_transformed[:, 0]
    info["y"] = X_transformed[:, 1]
    all_data = all_data.reset_index(drop=True)
    info["class"] = all_data["class"]
    info["dataset"] = all_data["dataset"]

    info.to_csv("PACMAP_coords_vaginal.csv", index=False)

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


def fig_3_6_main(args):
    france_path = args.france_data
    hickey_path = args.hickey_data
    hyuhn_path = args.hyuhn_data
    baker_path = args.baker_data

    france_data = pd.read_csv(france_path)
    hickey_data = pd.read_csv(hickey_path)
    hyuhn_data = pd.read_csv(hyuhn_path)
    baker_data = pd.read_csv(baker_path)

    france_train = pd.read_csv(args.france_train)
    france_test = pd.read_csv(args.france_test)
    france_validate = pd.read_csv(args.france_validate)

    if args.fig == None:
        fig = 8 if "clustered" in args.france_train else 3
    else: fig = args.fig

    fig_3(france_train, france_test, france_validate, figure=fig)
    # plot_pacmaps(france_data, hickey_data, hyuhn_data, baker_data, figure=6)

def fig_4_main(args):
    france_path = args.france_data
    hickey_path = args.hickey_data
    common_cols = args.common_cols.split(",")

    france_data = pd.read_csv(france_path)
    hickey_data = pd.read_csv(hickey_path)

    fig = 9 if "manghi" in france_path else 4

    plot_hickey_valencia_comparison(france_data, hickey_data, common_cols, figure=fig)
    get_pacmap_csv(france_data, hickey_data, common_cols)

def fig_6_main(args):
    stratabionn_60_path = args.stratabionn_class_60
    stratabionn_80_path = args.stratabionn_class_80
    validation_60_path = args.validation_60
    validation_80_path = args.validation_80

    validation_60_data = list(load_class(validation_60_path, class_lbl="HC_subCST")["subCST"])
    validation_80_data = list(load_class(validation_80_path, class_lbl="HC_subCST")["subCST"])

    stratabionn_60_data = list(load_class(stratabionn_60_path)["subCST"])
    stratabionn_80_data = list(load_class(stratabionn_80_path)["subCST"])

    plot_bars_oral(validation_60_data, validation_80_data, stratabionn_60_data, stratabionn_80_data)
    plot_confusion_oral(validation_80_data, stratabionn_80_data)

def fig_3_study_pacmap_main(args):
    base_class = args.base_class
    test_1_class = args.test_1_class
    test_2_class = args.test_2_class
    common_cols = args.common_cols.split(",")

    base_class_data = pd.read_csv(base_class)
    test_1_class_data = pd.read_csv(test_1_class)
    test_2_class_data = pd.read_csv(test_2_class)

    plot_3_study_classifications(base_class_data, test_1_class_data, test_2_class_data, common_cols, "3_study_")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots various graphs")

    parser.add_argument("--fig", help="name of figure", type=str)

    subparsers = parser.add_subparsers(dest='subcommand', help='Available subcommands')

    parser_fig_1_2 = subparsers.add_parser('fig_1_and_2', help='Figure 1 and 2 help')
    parser_fig_3_6 = subparsers.add_parser('fig_3_6', help='Figure 3 help')
    parser_fig_4 = subparsers.add_parser('fig_4', help='Figure 4 help')
    parser_fig_6 = subparsers.add_parser('fig_6', help='Figure 5 help')
    parser_fig_3_study_pacmap = subparsers.add_parser('3_study_pacmap', help='3_study_pacmap help')

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

    # Argument group for figure 3/6
    fig_3_6_group = parser_fig_3_6.add_argument_group('Fig 3 Options')
    fig_3_6_group.add_argument("--france-data", help="Path to France et al. data (VALENCIA formatted)", type=str)
    fig_3_6_group.add_argument("--hickey-data", help="Path to Hickey et al. data (VALENCIA formatted)", type=str)
    fig_3_6_group.add_argument("--hyuhn-data", help="Path to Hyuhn et al. data (VALENCIA formatted)", type=str)
    fig_3_6_group.add_argument("--baker-data", help="Path to Baker et al. data (VALENCIA formatted)", type=str)
    fig_3_6_group.add_argument("--france-train", help="Path to training set of France data", type=str)
    fig_3_6_group.add_argument("--france-test", help="Path to testing set of France data", type=str)
    fig_3_6_group.add_argument("--france-validate", help="Path to validation set of France data", type=str)

    # Argument group for figure 4
    fig_4_group = parser_fig_4.add_argument_group('Fig 4 Options')
    fig_4_group.add_argument("--france-data", help="Path to classified France et al. data", type=str)
    fig_4_group.add_argument("--hickey-data", help="Path to classified Hickey et al. data", type=str)
    fig_4_group.add_argument("--common_cols", help="Comma seperated common columns between france and hickey data", type=str)

    # Argument group for figure 4
    fig_6_group = parser_fig_6.add_argument_group('Fig 5 Options')
    fig_6_group.add_argument("--stratabionn-class-60", help="Classifications from Stratabionn 60/20/20", type=str)
    fig_6_group.add_argument("--stratabionn-class-80", help="Classifications from Stratabionn 80/10/10", type=str)
    fig_6_group.add_argument("--validation-60", help="Classifications from 60% validaiton set", type=str)
    fig_6_group.add_argument("--validation-80", help="Classifications from 80% validaiton set", type=str)

    # Argument group for figure 4
    fig_3_study_pacmap = parser_fig_3_study_pacmap.add_argument_group('3_study_pacmap Options')
    fig_3_study_pacmap.add_argument("--base-class", help="Base study classified", type=str)
    fig_3_study_pacmap.add_argument("--test-1-class", help="Classifications from Stratabionn 80/10/10", type=str)
    fig_3_study_pacmap.add_argument("--test-2-class", help="Classifications from 60% validaiton set", type=str)
    fig_3_study_pacmap.add_argument("--common_cols", help="Comma seperated common columns", type=str)


    # Parse arguments
    args = parser.parse_args()

    if args.subcommand == 'fig_1_and_2':
        fig_1_2_main(args)
    elif args.subcommand == 'fig_3_6':
        fig_3_6_main(args)
    elif args.subcommand == 'fig_4':
        fig_4_main(args)
    elif args.subcommand == 'fig_6':
        fig_6_main(args)
    elif args.subcommand == '3_study_pacmap':
        fig_3_study_pacmap_main(args)
    else:
        parser.print_help()
