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
from loader          import *
from naive_bayes     import *

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
    print_cmat(df[class_feature],labels,heatmap=True)
    return kmeans, labels

def kmeans_unsup_train_test(df,class_feature,nb_clusters=10):
    print("Running KMeans unsupervised clustering with {1} clusters on dataframe '{1}'".format(class_feature,nb_clusters,df.name))
    # Definition of the clusters
    cols = [col for col in df.columns if col!=class_feature]
    dataset_without_fc = df[cols]
    kmeans = KMeans(n_clusters = nb_clusters)
    clusters = kmeans.fit_predict(dataset_without_fc)
    print("Shape of the clusters centers: {0}".format(kmeans.cluster_centers_.shape))

    # Association of the clusters with their predicted label
    labels = np.zeros_like(clusters)
    for i in range(nb_clusters):
        mask = (clusters == i)
        labels[mask] = mode(df['label'][mask])[0]

    # Accuracy score:
    print("KMeans accuracy score: {0}".format(accuracy_score(df[class_feature], labels)))
    # Confusion matrix
    print_cmat(df[class_feature],labels,heatmap=True)
    return kmeans,labels



if __name__ == "__main__":
    # Loading and Preprocessing
    signs, signs_rd = load_base_dataset(path_x_train,path_y_train)
    sm_signs = select_instances(signs,'label')
    sm_signs_rd = randomise(sm_signs)

    ## UNCOMMENT IF YOU HAVE TO GENERATE THE FILES WITH THE BEST ATTRIBUTES
    # store_best_attributes()

    signs_ba2, signs_ba2_rd = dataset_best_n_attributes(2,signs)
    signs_ba5, signs_ba5_rd = dataset_best_n_attributes(5,signs)
    signs_ba10, signs_ba10_rd = dataset_best_n_attributes(10,signs)
    sm_signs_ba2, sm_signs_ba2_rd = dataset_best_n_attributes(2,sm_signs)
    sm_signs_ba5, sm_signs_ba5_rd = dataset_best_n_attributes(5,sm_signs)
    sm_signs_ba10, sm_signs_ba10_rd = dataset_best_n_attributes(10,sm_signs)

    # Store the datasets
    # signs_ba2.to_csv('./data/x_train_gr_smpl_2ba.csv')
    # signs_ba5.to_csv('./data/x_train_gr_smpl_5ba.csv')
    # signs_ba10.to_csv('./data/x_train_gr_smpl_10ba.csv')

    df_to_test = [
        # signs,
        # signs_rd,
        # signs_ba2_rd,
        # signs_ba5_rd,
        # signs_ba10_rd,
        # sm_signs,
        # sm_signs_rd,
        sm_signs_ba2_rd,
        sm_signs_ba5_rd,
        sm_signs_ba10_rd
    ]

    # Elbow method:
    # plot_elbow(sm_signs_rd)

    # Kmeans definition
    n_clusters_elbow = 3
    n_clusters = 10

    kmeans_sup, labels_sup = kmeans_sup_train_test(sm_signs_ba5_rd,'label')
    kmeans_unsup, labels_unsup = kmeans_unsup_train_test(sm_signs_ba5_rd,'label')

    # plt.figure()
    # cmat = confusion_matrix(signs_ba10_rd['label'], labels)
    # sns.heatmap(cmat.T, square=True, annot=True, fmt='d', cbar=False,
    #             xticklabels=signs_ba10_rd['label'],
    #             yticklabels=signs_ba10_rd['label'])
    # plt.xlabel('true label')
    # plt.ylabel('predicted label');
    # plt.show()

    # Kmeans visualisation
    # fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    # centers = kmeans_res.cluster_centers_.reshape(10, 48, 48)
    # for axi, center in zip(ax.flat, centers):
    #     axi.set(xticks=[], yticks=[])
    #     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    # plt.show()
