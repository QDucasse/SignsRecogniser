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
from sklearn.cluster import KMeans

# ============================================
#            K-MEANS CLUSTERING
# ============================================

# Unsupervised -> Drop 'label' field
signs_ba2_rd_nolab = signs_ba2_rd.drop('label',axis = 1)
signs_ba2_rd_std = normalize_dataset(signs_ba2_rd)

signs_ba5_rd_nolab = signs_ba5_rd.drop('label',axis = 1)
signs_ba5_rd_std = normalize_dataset(signs_ba5_rd)

signs_ba10_rd_nolab = signs_ba10_rd.drop('label',axis = 1)
signs_ba10_rd_std = normalize_dataset(signs_ba10_rd)

def plot_elbow(df,nb_clusters = 11):
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

# plot_elbow(signs_ba2_rd_std)
# # found 3/4 clusters
# plot_elbow(signs_ba5_rd_std)
# # found 3/4 clusters
# plot_elbow(signs_ba10_rd_std)
# # found 3/4 clusters

n_clusters_elbow = 3
n_clusters = 10

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

kmeans_cluster = KMeans(n_clusters = 10)
kmeans_cluster.fit(signs_ba5_rd_std)
print(kmeans_cluster.labels_[::10])
print("KMeans accuracy : ", accuracy_score(signs_ba5_rd['label'],kmeans_cluster.labels_, normalize = True))
