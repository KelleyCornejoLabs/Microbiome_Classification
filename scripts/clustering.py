# Alexander Symons | Sept 12 2023 |  clustering.py
# Very rough script that can show why kmeans is best,
# and why clustering normalized data is a bad idea

import sklearn.cluster
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nn_classifier import load_data
import matplotlib.cm as cm

X_train, y_train, X_test, y_test, all_labels, ordered_prevelence, count_columns = load_data("nn_training_train.csv", "nn_training_test.csv", norm="none")

print(X_train.shape)

X_train = X_train.cpu().numpy().astype("float")
y_train = y_train.cpu().argmax(dim=1).numpy().astype("float")

centroids = pd.read_csv("new_centroids_subCST.csv")

lbls = centroids["sub_CST"].to_numpy()
data = centroids.drop("sub_CST", axis=1).to_numpy().astype("float")

clf = sklearn.neighbors.NearestCentroid()
clf.fit(X_train, y_train)

kmeans = sklearn.cluster.KMeans(n_clusters=3, n_init="auto").fit(X_train)
spectral = sklearn.cluster.SpectralClustering(n_clusters=3)
dbscan = sklearn.cluster.DBSCAN(eps=0.15)
hdbscan = sklearn.cluster.HDBSCAN(min_cluster_size=100)

"""X_train_centroids = np.concatenate((X_train, data))
print("same?", not False in np.equal(X_train_centroids[-1:-14:-1], data[::-1]))
db_preds = dbscan.fit_predict(X_train)
print("DBSCAN", db_preds[-1:-14:-1][::-1])

acc = lambda x,y: [a == b for a,b in zip(x, y)].count(True)/len(y)

preds = kmeans.predict(X_train_centroids)
preds_train = kmeans.predict(X_train)
print("K-Means", len(data), preds[-1:-14:-1][::-1])

clfpreds = clf.predict(X_train_centroids)
clf_train_preds = clf.predict(X_train)
print("Nearest Centroid", clfpreds[-1:-14:-1][::-1], len(clf.centroids_), acc(clf_train_preds, y_train))

preds = spectral.fit_predict(X_train_centroids)
print("Spectral", preds[-1:-14:-1][::-1])"""

def score(data, lbls):
    print(f"Calinski (higher better): {sklearn.metrics.calinski_harabasz_score(data, lbls)}")
    print(f"Davies (closer to 0 better): {sklearn.metrics.davies_bouldin_score(data, lbls)}")
    print(f"Silhouette (higher better): {sklearn.metrics.silhouette_score(data, lbls)}")


preds_train = kmeans.predict(X_train)
print("\nKmeans")
score(X_train, preds_train)

db_preds = dbscan.fit_predict(X_train)
print("\nDBSCAN")
score(X_train, db_preds)

hdb_preds = hdbscan.fit_predict(X_train)
print("\nHDBSCAN")
score(X_train, hdb_preds)

silhouettes = []

for i in range(1,200):
    print(i)
    s = lambda x:sklearn.metrics.silhouette_score(X_train, x)
    hdbscan = sklearn.cluster.HDBSCAN(min_cluster_size=100)
    hdb_preds = hdbscan.fit_predict(X_train)
    silhouettes.append(s(hdb_preds))

plt.plot(range(1,200), silhouettes)
plt.show()

#s_preds = spectral.fit_predict(X_train)
#print("\nSpectral")
#score(X_train, s_preds)

exit()

kmean_cal_scores = []
spectral_cal_scores = []
kmean_dav_scores = []
spectral_dav_scores = []
kmean_sil_scores = []
spectral_sil_scores = []


"""for n in range(2,25):
    print(n)
    s = lambda x:sklearn.metrics.silhouette_score(X_train, x)
    d = lambda x:sklearn.metrics.davies_bouldin_score(X_train, x)
    c = lambda x:sklearn.metrics.calinski_harabasz_score(X_train, x)

    kmeans = sklearn.cluster.KMeans(n_clusters=n, n_init="auto").fit(X_train)
    spectral = sklearn.cluster.SpectralClustering(n_clusters=n)

    preds_train = kmeans.predict(X_train)
    kmean_sil_scores.append(s(preds_train))
    kmean_cal_scores.append(c(preds_train))
    kmean_dav_scores.append(d(preds_train))

    s_preds = spectral.fit_predict(X_train)
    spectral_sil_scores.append(s(s_preds))
    spectral_cal_scores.append(c(s_preds))
    spectral_dav_scores.append(d(s_preds))


plt.plot(range(2,25), kmean_cal_scores)
plt.plot(range(2,25), spectral_cal_scores)
plt.legend(["KMeans - cal", "Spectral - cal"])
plt.show()

plt.plot(range(2,25), kmean_dav_scores)
plt.plot(range(2,25), spectral_dav_scores)
plt.legend(["KMeans - dav", "Spectral - dav"])
plt.show()

plt.plot(range(2,25), kmean_sil_scores)
plt.plot(range(2,25), spectral_sil_scores)
plt.legend(["KMeans - sil", "Spectral - sil"])
plt.show()"""


range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_train) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    #clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    #clusterer = sklearn.cluster.SpectralClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X_train)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = sklearn.metrics.silhouette_score(X_train, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = sklearn.metrics.silhouette_samples(X_train, cluster_labels)

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
        X_train[:, 0], X_train[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
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

plt.show()