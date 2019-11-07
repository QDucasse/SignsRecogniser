'''
# created 15/10/2019 14:22
# by Q.Ducasse
'''

import cv2
import sklearn
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
from scipy.stats     import mode
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from loader          import divide_by_255, load_base_dataset, path_x_train, path_y_train
from naive_bayes     import dataset_best_n_attributes

# ============================================
#            K-MEANS CLUSTERING
# ============================================

def plot_elbow(df,nb_clusters = 11):
    '''
    Perform K-Means algorithm with a certain number of clusters and
    print the wcss (Within Clusters Sum of Squares) in order to perform
    the "elbow method".
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset that will run through the algorithm.
    nb_clusters: int
        Maximal number of clusters the algorithm will have to use.
    '''
    wcss = []
    for i in range(1, nb_clusters):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def unsup_kmeans_train_test(df,df_std,n_clusters):
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(df_std)
    #beginning of  the cluster numbering with 1 instead of 0
    y_kmeans1=y_kmeans
    y_kmeans1=y_kmeans+1
    # New Dataframe called cluster
    cluster = pd.DataFrame(y_kmeans1)
    # Adding cluster to the Dataset1
    df['cluster'] = cluster
    #Mean of clusters
    kmeans_mean_cluster = pd.DataFrame(round(df.groupby('cluster').mean(),1))
    return kmeans_mean_cluster

if __name__ == "__main__":
    # Loading and Preprocessing
    signs, signs_rd = load_base_dataset(path_x_train,path_y_train)
    signs = divide_by_255(signs,'label')
    signs_rd = divide_by_255(signs_rd,'label')

    signs_ba2,signs_ba2_rd   = dataset_best_n_attributes(2,signs)
    signs_ba5,signs_ba5_rd   = dataset_best_n_attributes(5,signs)
    signs_ba10,signs_ba10_rd = dataset_best_n_attributes(10,signs)

    # Removing the class feature in order to obtain an unsupervised prediction
    signs_rd_nolab      = signs_rd.drop('label',axis = 1)
    signs_ba2_rd_nolab  = signs_ba2_rd.drop('label',axis = 1)
    signs_ba5_rd_nolab  = signs_ba5_rd.drop('label',axis = 1)
    signs_ba10_rd_nolab = signs_ba10_rd.drop('label',axis = 1)

    # Elbow method:
    # plot_elbow(signs_ba2_rd_std)
    # # found 3/4 clusters
    # plot_elbow(signs_ba5_rd_std)
    # # found 3/4 clusters
    # plot_elbow(signs_ba10_rd_std)
    # # found 3/4 clusters

    # Kmeans definition
    n_clusters_elbow = 3
    n_clusters = 10
    kmeans = KMeans(n_clusters = 10)
    clusters = kmeans.fit_predict(signs_ba5_rd_nolab)
    print(kmeans.cluster_centers_.shape)

    # Kmeans visualisation
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    centers = kmeans.cluster_centers_.reshape(10, 10, 5)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    plt.show()

    labels = np.zeros_like(clusters)
    for i in range(10):
        mask = (clusters == i)
        labels[mask] = mode(signs_ba5_rd['label'][mask])[0]

    print(accuracy_score(signs_ba5_rd['label'], labels))

    #print("KMeans accuracy : ", accuracy_score(signs_ba5_rd['label'],kmeans_cluster.labels_, normalize = True))
