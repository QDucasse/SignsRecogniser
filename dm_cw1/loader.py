'''
# created 06/10/2019 14:22
# by Q.Ducasse
'''

import cv2
import sklearn
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix

# ============================================
#       CSV FILE LOADING AND VISUALISATION
# ============================================

## Paths creation and generation
## =============================

path_x_train = './data/x_train_gr_smpl.csv'
path_y_train = './data/y_train_smpl.csv'

## Data import & Preprocessing
## ===========================

# Import images vector values
signs = pd.read_csv(path_x_train, sep=',',na_values='None')
# Import labels
labels = pd.read_csv(path_y_train, sep=',',na_values='None')

# Link label to vectors
signs.insert(0,"label",labels.values)

# Line Randomisation
signs_rd = signs.sample(frac=1)
# Optional: Save again as CSV:
# signs_rd.to_csv('./data/x_train_gr_smpl_rd.csv')

# One label only data sets
# ========================

def create_one_label_dataset(nb):
    '''
    Return a dataset with only the labels nb
    '''
    label_is_correct = (signs['label'] == nb)
    signs_1label = signs[label_is_correct]
    path = './data/signs_label' + str(nb) + '.csv'
    signs_1label.to_csv(path)

def load_one_label_dataset(sample_nb):
    '''
    Load a dataset generated by create_one_label_dataset
    '''
    if ((sample_nb < 0) or (sample_nb > 9)):
        raise Exception('No data sample with that number')
    # Load the csv dataset
    dataframe = pd.read_csv('./data/signs_label' + str(sample_nb) + '.csv')
    # Drop the index column
    dataframe = dataframe.drop('Unnamed: 0', axis = 1)
    return dataframe

def dataset_with_boolean_mask(nb):
    '''
    Return the signs dataframe with labels formatted as:
        0 if label == nb
        1 if label != nb

        NOT WORKING !!!
    '''
    signs_1label = signs
    signs_1label.loc[signs_1label['label'] == nb, ['label']] = '0'
    signs_1label.loc[signs_1label['label'] != nb, ['label']] = '1'
    return signs_1label

def load_boolean_mask(nb):
    path_x_train = './data/x_train_gr_smpl.csv'
    path_labels  = './data/y_train_smpl_' + str(nb) + '.csv'
    signs_1label = pd.read_csv(path_x_train, sep=',',na_values='None')
    labels = pd.read_csv(path_labels, sep=',',na_values='None')
    signs_1label.insert(0,"label",labels.values)
    return signs_1label


## Data visualisation
## ==================
# Check the first and last rows of the basic + randomised data set
# print(signs.head(n=5))
# print(signs.tail(n=5))
# print(signs_rd.head(n=5))
# print(signs_rd.tail(n=5))
#print(signs['0'])

# Display the number of labels of each kind in the dataset
# plt.figure()
# sns.set(style="whitegrid", color_codes=True)
# sns.countplot('label',data=signs_rd)
# plt.show()


## Prepare training/test data
## ==========================

# Separation between feature vector (image itself) and target (label)
def separate_train_test(label_class, ratio=0.20):
    cols   = [col for col in signs_rd.columns if col!=label_class]
    data   = signs_rd[cols]
    target = signs_rd[label_class]

    # Separation between training/test data with the given ratio
    data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = ratio, random_state = 10)
    return data_train, data_test, target_train, target_test

data_train, data_test, target_train, target_test = separate_train_test('label')

# ============================================
#            NAIVE BAYES BENCHMARK
# ============================================

# # import the necessary module
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics     import accuracy_score
# #create an object of the type GaussianNB
# gnb = GaussianNB()
# #train the algorithm on training data and predict using the testing data
# pred = gnb.fit(data_train, target_train).predict(data_test)
# #print the accuracy score of the model
# print("Naive-Bayes accuracy : ", accuracy_score(target_test, pred, normalize = True))
#
# # create the confusion matrix of the model
# # with numpy
# bayes_conf_mat_np = confusion_matrix(target_test,pred)
# # with pandas
# bayes_conf_mat_pd = pd.crosstab(target_test,pred)
#
# # print(bayes_conf_mat_pd)

# ============================================
#           CORRELATION MATRIX
# ============================================

# Using Pearson Correlation

def store_best_abs_corr_attributes(nb):
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

# Print heatmap
# plt.figure(figsize=(12,10))
# print(signs_label3['label'].corr(signs_label3['0']))
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

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
    ba_n = ba0[:nb] + ba1[:nb] + ba2[:nb] + ba3[:nb] + ba4[:nb] \
            + ba5[:nb] + ba6[:nb] + ba7[:nb] + ba8[:nb] + ba9[:nb]
    ba_n = [str(i) for i in ba_n]
    return ba_n

ba_n2  = best_n_attributes(2)
ba_n5  = best_n_attributes(5)
ba_n10 = best_n_attributes(10)

signs_ba2  = signs[ba_n2]
print(signs_ba2)
signs_ba2.to_csv('./data/x_train_gr_smpl_2ba.csv')
signs_ba5  = signs[ba_n5]
print(signs_ba5)
signs_ba5.to_csv('./data/x_train_gr_smpl_5ba.csv')
signs_ba10 = signs[ba_n10]
print(signs_ba10)
signs_ba10.to_csv('./data/x_train_gr_smpl_10ba.csv')
