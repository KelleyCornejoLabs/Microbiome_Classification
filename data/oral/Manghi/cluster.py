import pandas as pd
import argparse
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score, silhouette_samples

def extract_numpy(df: pd.DataFrame, norm:bool = True, id_col:str = "sampleID") -> np.array:
    normalized_data = df.drop(columns=[id_col]).astype(float)
    
    if norm:
        normalized_data = normalized_data.div(normalized_data.sum(axis=1), axis=0)
        normalized_data[normalized_data.isnull()] = 1.0e-5
        normalized_data[normalized_data.eq(0)] = 1.0e-5

    return normalized_data.to_numpy()

def print_elbow(data: np.array, n: int):
    inertias = []
    for n_clusters in range(1,n):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        inertias.append(-kmeans.score(data))

    plt.plot(range(1,n), inertias)
    plt.savefig("elbow.jpeg")
    plt.clf()

def print_silhouettes(data: np.array, n: int):
    scores = []
    for n_clusters in range(2,n):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        scores.append(silhouette_score(data, kmeans.labels_))

    plt.plot(range(2,n), scores)
    plt.savefig("silhouette.jpeg")
    plt.clf()

def plot_silhouette(data: np.array, n: int):
    for n_clusters in range(2,n):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            data[:, 0], data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        plt.savefig(f"silhouette_{n_clusters}.jpeg")
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains or uses a neural network classifier on microbiome count data")

    # Arguments for tool
    arguments = parser.add_argument_group("Arguments")
    arguments.add_argument("-in", "--input", help="Path to train input data as csv", default=None, required = True)
    arguments.add_argument("-out", "--output", help="Path to train input data as csv", default="clsutered.csv")
    arguments.add_argument("-cl", "--clusters", type=int, help="Number of clusters to use", default=None)
    arguments.add_argument("-gr", "--graph", action=argparse.BooleanOptionalAction, help="Generate graphs of sihlouette and inertia scores", default=False)
    args = parser.parse_args()

    path = args.input
    out = args.output

    df = pd.read_csv(path).drop(columns=["HC_subCST"])

    # Normalize data
    data = extract_numpy(df, norm = True)
    
    if args.graph:
        print_elbow(data, 10)
        plot_silhouette(data, 10)
        print_silhouettes(data, 10)
        exit(0)

    if args.clusters == None:
        print("ERROR: --clusters <int> required for clustering")
        exit(1)

    kmeans = KMeans(n_clusters=args.clusters)
    predictions = kmeans.fit_predict(data)

    new_column_names = [col[:-1] if col.endswith(" ") else col for col in df.columns]
    df = df.rename(columns={old:new for old, new in zip(df.columns, new_column_names)})

    unique_csts = [f"CST{i}" for i in range(args.clusters)]
    df["HC_subCST"] = list(map(lambda x:unique_csts[x], predictions))

    df.to_csv(out, index=False)