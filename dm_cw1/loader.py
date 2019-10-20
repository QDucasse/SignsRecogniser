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
    label_is_correct = (signs['label'] == nb)
    signs_1label = signs[label_is_correct]
    path = './data/signs_label' + str(nb) + '.csv'
    signs_1label.to_csv(path)
#
# for i in range(10):
#     create_one_label_dataset(i)
#
def load_one_label_dataset(sample_nb):
    if ((sample_nb < 0) or (sample_nb > 9)):
        raise Exception('No data sample with that number')
    # Load the csv dataset
    dataframe = pd.read_csv('./data/signs_label' + str(sample_nb) + '.csv')
    # Drop the index column
    dataframe = dataframe.drop('Unnamed: 0', axis = 1)
    return dataframe

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

# import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics     import accuracy_score
#create an object of the type GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
#print the accuracy score of the model
print("Naive-Bayes accuracy : ", accuracy_score(target_test, pred, normalize = True))

# ============================================
#           CORRELATION MATRIX
# ============================================

#Using Pearson Correlation

signs_label3 = load_one_label_dataset(3)
print(signs_label3.head(5))

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = cor = signs_label3.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor["label"])
print(cor)
relevant_features = cor_target[cor_target>0.3]
print(relevant_features)
