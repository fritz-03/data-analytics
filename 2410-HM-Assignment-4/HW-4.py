import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def k_means_clustering():
    try:
        k = 10
        df = pd.read_excel('iris.xlsx', sheet_name=f'Sheet1')
        print(f'The length(rows) of dataset: {len(df)}')

        x = df.iloc[:, 0:4].values
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(x)
        print(f'Running with the k value of {k}')

        print('The cluster centroids:')
        for centroid in kmeans.cluster_centers_:
            formatted_centroid = [f'{coord:.2f}' for coord in centroid]
            print(formatted_centroid)

        print('The Sample Clusters:')
        print(kmeans.labels_)

        # Define colors and symbols for plotting
        colors = {0: 'orange', 1: 'blue', 2: 'green'}
        symbols = {0: '+', 1: 'o', 2: '^'}
        cluster_species = {0: 'Iris-versicolor', 1: 'Iris-setosa', 2: 'Iris-virginica'}

        # Define pairs of columns to create cluster plots
        column_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        column_names = {0: 'Sepal length', 1: 'Sepal width', 2: 'Petal length', 3: 'Petal width'}

        for pair in column_pairs:
            x_label, y_label = pair
            plt.figure()
            for i in range(3):
                cluster_points = x[kmeans.labels_ == i]
                plt.scatter(cluster_points[:, x_label], cluster_points[:, y_label], c=colors[i], marker=symbols[i],
                            label=f'{cluster_species[i]}')

            plt.scatter(kmeans.cluster_centers_[:, x_label], kmeans.cluster_centers_[:, y_label], c='red', marker='x',
                        label='Centroids')

            plt.xlabel(column_names[x_label])
            plt.ylabel(column_names[y_label])
            plt.title(f'K-means clustering - Iris data points and cluster centroids')
            plt.legend()
            plt.show()

        print(f'The silhouette_score: {silhouette_score(x, kmeans.labels_):.4f}')
    except Exception as e:
        print(e)


def elbow_method():
    try:
        df = pd.read_excel('iris.xlsx', sheet_name=f'Sheet1')
        x = df.iloc[:, 0:4].values
        k_range = range(1, 11)
        mean_distortions = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(x)
            mean_distortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_,
                                                     'euclidean'), axis=1) / x.shape[0]))

        plt.plot(k_range, mean_distortions, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Average Distortion')
        plt.title('Selecting k with the Elbow Method for iris dataset')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(e)


def main():
    k_means_clustering()
    elbow_method()


if __name__ == '__main__':
    main()
