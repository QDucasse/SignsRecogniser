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
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.cluster import KMeans
from loader          import divide_by_255, load_base_dataset, path_x_train, path_y_train
from naive_bayes     import dataset_best_n_attributes, print_cmat

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

def df_no_class(df,class_feature):
    df_no_class = df.drop(class_feature,axis = 1)
    df_no_class.name = df.name + " no class feature"
    return df_no_class


def kmeans_sup_train_test(df,class_feature,nb_clusters=10):
    print("Running KMeans supervised clustering (on feature {0}) with {1} clusters on dataframe '{2}'".format(class_feature,nb_clusters,df.name))
    # Definition of the clusters
    #TODO! Separate train/test data
    kmeans = KMeans(n_clusters = nb_clusters)
    clusters = kmeans.fit_predict(df)
    print("Shape of the clusters centers: {0}".format(kmeans.cluster_centers_.shape))

    # Association of the clusters with their predicted label
    labels = np.zeros_like(clusters)
    for i in range(nb_clusters):
        mask = (clusters == i)
        labels[mask] = mode(df['label'][mask])[0]

    # Accuracy score:
    print(accuracy_score(df[class_feature], labels))
    # Confusion matrix
    # print_cmat(df[class_feature],labels)

    # Print heatmap

    return kmeans, labels

def kmeans_unsup_train_test(df,df_nolab,class_feature,nb_clusters=10):
    print("Running KMeans unsupervised clustering with {1} clusters on dataframe '{1}'".format(class_feature,nb_clusters,df.name))
    # Definition of the clusters
    #TODO! Separate train/test data
    kmeans = KMeans(n_clusters = nb_clusters)
    clusters = kmeans.fit_predict(df_nolab)
    print("Shape of the clusters centers: {0}".format(kmeans.cluster_centers_.shape))

    # Association of the clusters with their predicted label
    labels = np.zeros_like(clusters)
    for i in range(nb_clusters):
        mask = (clusters == i)
        labels[mask] = mode(df['label'][mask])[0]

    # Accuracy score:
    print(accuracy_score(df[class_feature], labels))
    # Confusion matrix
    # print_cmat(df[class_feature],labels)
    return kmeans

def kmeans_visualisation(kmeans,shape,nb_clusters=10):
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    centers = kmeans.cluster_centers_.reshape(10, 48, 48)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    plt.show()


if __name__ == "__main__":
    # Loading and Preprocessing
    signs, signs_rd = load_base_dataset(path_x_train,path_y_train)
    signs = divide_by_255(signs,'label')
    signs_rd = divide_by_255(signs_rd,'label')

    signs_ba2,signs_ba2_rd   = dataset_best_n_attributes(2,signs)
    signs_ba5,signs_ba5_rd   = dataset_best_n_attributes(5,signs)
    signs_ba10,signs_ba10_rd = dataset_best_n_attributes(10,signs)

    # Removing the class feature in order to obtain an unsupervised prediction
    signs_rd_nolab      = df_no_class(signs_rd,'label')
    signs_ba2_rd_nolab  = df_no_class(signs_ba2_rd,'label')
    signs_ba5_rd_nolab  = df_no_class(signs_ba5_rd,'label')
    signs_ba10_rd_nolab = df_no_class(signs_ba10_rd,'label')

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

    kmeans_res, labels = kmeans_sup_train_test(signs_ba10_rd,'label')

    plt.figure()
    cmat = confusion_matrix(signs_ba10_rd['label'], labels)
    sns.heatmap(cmat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=signs_ba10_rd['label'],
                yticklabels=signs_ba10_rd['label'])
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    plt.show()

    # Kmeans visualisation
    # fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    # centers = kmeans_res.cluster_centers_.reshape(10, 48, 48)
    # for axi, center in zip(ax.flat, centers):
    #     axi.set(xticks=[], yticks=[])
    #     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    # plt.show()
