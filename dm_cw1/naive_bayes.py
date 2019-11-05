'''
# created 10/10/2019 13:54
# by Q.Ducasse
'''
import cv2
import sklearn
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
from sklearn.metrics         import confusion_matrix,accuracy_score
from sklearn.naive_bayes     import GaussianNB
from loader import load_base_dataset, path_x_train, path_y_train


# ============================================
#            NAIVE BAYES BENCHMARK
# ============================================

def gaussian_train_test(df,class_feature):
    print("Running Naive Bayes on dataframe '{0}' with class feature '{1}'".format(df.name,class_feature))
    # separate train and test data
    data_train, data_test, target_train, target_test = separate_train_test(df,class_feature)
    # create an object of the type GaussianNB
    gnb = GaussianNB()
    #train the algorithm on training data and predict using the testing data
    pred = gnb.fit(data_train, target_train).predict(data_test)
    #print the accuracy score of the model
    print("Naive-Bayes accuracy : ", accuracy_score(target_test, pred, normalize = True))
    # create the confusion matrix of the model
    print_cmat(target_test,pred)

def print_cmat(actual,pred):
    print(pd.crosstab(actual,pred,rownames=['Actual'],colnames=['Predicted']))


# ============================================
#           CORRELATION MATRIX
# ============================================

# Using Pearson Correlation

def store_best_abs_corr_attributes(nb,cor_score = 0.01):
    signs_1label = load_boolean_mask(nb)
    cor = signs_1label.corr()
    cor_target = abs(cor["label"])
    relevant_features = cor_target[cor_target>0.01].sort_values(ascending=False)
    relevant_features.to_csv('./data/best_features_smpl_' + str(nb) + '.csv')
    return relevant_features.head(n=10)

def store_best_attributes():
    for i in range(10):
        print("Finding best attributes for " + str(i))
        store_best_abs_corr_attributes(i)

# Print heatmap, not supported due to the number of attributes!
def print_heatmap(df):
    plt.figure(figsize=(12,10))
    print(df['label'].corr(df['0']))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

# From the created files
# ba = best atributes

# Hardcoded values
ba0 = [1172, 1171, 1468, 1220, 1472, 1221,
       1123, 1469, 1419, 1519, 1124, 1471]

ba1 = [1094, 1046, 1172, 1142,  998, 1190,
       1173,  997,  981, 1045,  950, 1143]

ba2 = [1084, 1132, 1083, 1131, 1082, 1130,
       1036, 1081, 1179, 1035, 1129, 1178]

ba3 = [1696, 1697, 1648, 1649, 1695, 1698,
       1745, 1744, 1713, 1712, 1647, 1650]

ba4 = [1849, 1850, 1848, 1897, 1801, 1802,
       1898, 1800, 1896, 1043, 1847, 1309]

ba5 = [1319, 1367, 1271, 1368, 1270, 1318,
       1320, 1415, 1416, 1222, 1366, 1223]

ba6 = [1186, 1738, 1934, 1737, 1786, 1787,
       1885, 1689, 1983, 1688,  633, 1836]

ba7 = [1168, 1120, 1169, 1119, 1167, 1121,
       1216, 1072, 1071, 1215, 1217, 1118]

ba8 = [1280, 1232, 1328, 1184, 1375, 1376,
       1327, 1470, 1423, 1422, 1471, 1469]

ba9 = [1362, 1410, 1363, 1364, 1411, 1365,
       1412, 1317, 1318, 1413, 1361, 1314]

# Create datasets with the n best features

def best_n_attributes(nb):
    global ba0,ba1,ba2,ba3,ba4,ba5,ba6,ba7,ba8,ba9
    ba_n = ['label'] + ba0[:nb] + ba1[:nb] + ba2[:nb] + ba3[:nb] + ba4[:nb] \
                     + ba5[:nb] + ba6[:nb] + ba7[:nb] + ba8[:nb] + ba9[:nb]
    ba_n = [str(i) for i in ba_n]
    return ba_n

def dataset_best_n_attributes(nb,df):
    ba_n = best_n_attributes(nb)
    # Filter the dataset
    df_ba_n = df[ba_n]
    df_ba_n.name = "Dataframe with best " + str(nb) + " attributes"
    # Randomize the dataset
    df_ba_n_rd = df_ba_n.sample(frac=1)
    df_ba_n_rd.name = "Dataframe with best " + str(nb) + " attributes randomized"
    return df_ba_n,df_ba_n_rd

# Create the datasets
signs, signs_rd = load_base_dataset(path_x_train,path_y_train)
signs_ba2,signs_ba2_rd = dataset_best_n_attributes(2,signs)
signs_ba5,signs_ba5_rd = dataset_best_n_attributes(5,signs)
signs_ba10,signs_ba10_rd = dataset_best_n_attributes(10,signs)

# Print the datasets
# print(signs_ba2)
# print(signs_ba5)
# print(signs_ba10)
# Store the datasets
# signs_ba2.to_csv('./data/x_train_gr_smpl_2ba.csv')
# signs_ba5.to_csv('./data/x_train_gr_smpl_5ba.csv')
# signs_ba10.to_csv('./data/x_train_gr_smpl_10ba.csv')

# Run Bayes over the new sets
# gaussian_train_test(signs_ba2_rd,'label')
# gaussian_train_test(signs_ba5_rd,'label')
# gaussian_train_test(signs_ba10_rd,'label')
