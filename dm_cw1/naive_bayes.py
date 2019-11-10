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
from loader import *
from sklearn.metrics         import confusion_matrix,accuracy_score
from sklearn.naive_bayes     import GaussianNB

# ============================================
#            NAIVE BAYES BENCHMARK
# ============================================

def gaussian_train_test(df,class_feature,ratio=0.2,heatmap=False):
    '''
    Runs a gaussian model over the dataset with the class feature.

    '''
    print("================================================================")
    print("Running Naive Bayes on dataframe '{0}'\nwith class feature '{1}'".format(df.name,class_feature))
    # Separate train and test data
    data_train, data_test, target_train, target_test = separate_train_test(df,class_feature,ratio=ratio)
    # Create a Gaussian Naive Bayes model
    gnb = GaussianNB()
    # Train the algorithm on training data and predict using the testing data
    pred = gnb.fit(data_train, target_train).predict(data_test)
    # Print the accuracy score of the model
    print("Naive-Bayes accuracy: ", accuracy_score(target_test, pred, normalize = True))
    # Print the confusion matrix of the model
    print_cmat(target_test,pred,heatmap=heatmap)

def print_cmat(actual,pred,heatmap=False):
    '''
    Print the confusion matrix for the two provided values.
    Parameters
    ==========
    actual: Pandas.Dataframe
        Actual values obtained extracted from the dataset.
    pred: Pandas.Dataframe
        Actual values obtained through the fit() method of the model.
    '''
    cor = pd.crosstab(actual,pred,rownames=['Actual'],colnames=['Predicted'])
    print(cor)
    if heatmap:
        ax = sns.heatmap(cor,annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()


# ============================================
#           CORRELATION MATRIX
# ============================================

# Using Pearson Correlation

def best_cor_attributes(df,class_feature,cor_score = 0.01):
    '''
    Stores the n best attrivutes in terms of correlation with the class attribute
    for a given dataset.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset to use.
    class_feature: string
        Class feature to use for the correlation.
    cor_score: float
        Score that the correlation should obtain to be considered relevant.
        Default: 0.01
    Returns
    '''
    cor = df.corr()
    cor_target = abs(cor[class_feature])
    relevant_features = cor_target[cor_target>0.01].sort_values(ascending=False)
    return relevant_features

def store_best_attributes_for_label(nb,df,class_feature,cor_score = 0.01):
    '''
    Store the best attributes in the corresponding file.
    Parameters
    ==========
    nb: int
        Label to use.
    + Same parameters as best_cor_attributes().
    '''
    relevant_features = best_cor_attributes(nb,df,class_feature,cor_score)
    pd.to_csv(relevant_features, path_best_features(nb))

# signs_1label = load_boolean_mask(nb)
# store_best_attributes(signs_1label,2)

def store_best_attributes():
    '''
    Run the best feature selection process for the 10 labels.
    '''
    for i in range(10):
        print("Finding best attributes for " + str(i))
        signs_1label = load_boolean_mask(i)
        store_best_attributes_for_label(i,signs_1label,'label')

# Print heatmap, not supported due to the number of attributes!
def print_heatmap(df):
    '''
    Print the heatmap of the current dataframe.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset whose heatmap needs to be printed.
    '''
    plt.figure(figsize=(12,10))
    print(df['label'].corr(df['0']))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

# From the created files
# ba = best atributes

# Hardcoded values, extracted from the files.
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
    '''
    Select the n best attributes from the best attributes selection from all
    labels.
    Parameters
    ==========
    nb: int
        Number of attributes to select.

    Returns
    =======
    ba_n: [str] + int list
        List of the name of the columns for the best n attributes + 'label'.
        Should look like:
        ['label', best attribute 1 for label 0, best attribute 2 for label 0, ...
        best attribute n for label 0, ... , best attribute n for label 9]
    '''
    global ba0,ba1,ba2,ba3,ba4,ba5,ba6,ba7,ba8,ba9
    ba_n = ['label'] + ba0[:nb] + ba1[:nb] + ba2[:nb] + ba3[:nb] + ba4[:nb] \
                     + ba5[:nb] + ba6[:nb] + ba7[:nb] + ba8[:nb] + ba9[:nb]
    ba_n = [str(i) for i in ba_n]
    return ba_n

def display_nth_ba(n):
    '''
    Display the nth sign of the dataset.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset from which the element will be taken.
    n: int
        Row of the dataset.
    '''
    best_attributes = ba0[:n] + ba1[:n] + ba2[:n] + ba3[:n] + ba4[:n] \
                     + ba5[:n] + ba6[:n] + ba7[:n] + ba8[:n] + ba9[:n]
    visualize_image = []
    for i in range(2304):
        if (i in best_attributes):
            visualize_image.append(255)
        else:
            visualize_image.append(0)
    plt.figure()
    plt.imshow(np.array(visualize_image).reshape((48,48)),cmap='Greys')
    plt.show()


def dataset_best_n_attributes(nb,df):
    '''
    Generates a dataset and randomised version of the given dataset with the extracted
    n best attributes.
    Parameters
    ==========
    nb: int
        Number of best attributes to use.
    df: Pandas.Dataframe
        Dataset to use.
    '''
    ba_n = best_n_attributes(nb)
    # Filter the dataset
    df_ba_n = df[ba_n]
    df_ba_n.name =  df.name + " with best " + str(nb) + " attributes"
    # Randomize the dataset
    df_ba_n_rd = df_ba_n.sample(frac=1)
    df_ba_n_rd.name = df.name + " with best " + str(nb) + " attributes randomised"
    return df_ba_n,df_ba_n_rd

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
    # Run Bayes over the new sets
    for df in df_to_test:
        gaussian_train_test(df,'label',heatmap=False)

    # Display the position of the best attributes
    # display_nth_ba(2)
    # display_nth_ba(5)
    # display_nth_ba(10)
