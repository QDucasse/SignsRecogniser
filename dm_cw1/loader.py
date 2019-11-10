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
from sklearn                 import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix,accuracy_score
from sklearn.naive_bayes     import GaussianNB
from sklearn.cluster         import KMeans

# ============================================
#       CSV FILE LOADING AND VISUALISATION
# ============================================

## Paths creation and generation
## =============================

path_x_train = './data/x_train_gr_smpl.csv'
path_y_train = './data/y_train_smpl.csv'

def path_best_features(nb):
    '''
    Generate the path for the best features csv file.
    Parameters
    ==========
    nb: int
        Label of the wanted path.

    Returns
    path: string
        Path to the best features file for the label <nb>
    '''
    if(nb < 0 or nb > 9):
        raise Exception('No data sample with that number')
    return './data/best_features_smpl_' + str(nb) + '.csv'

def path_boolean_mask(nb):
    '''
    Generate the path for the boolean mask csv file.
    Parameters
    ==========
    nb: int
        Label of the wanted path.

    Returns
    path: string
        Path to the bolean mask file for the label <nb>
    '''
    if(nb < 0 or nb > 9):
        raise Exception('No data sample with that number')
    return './data/y_train_smpl_' + str(nb) + '.csv'

## Data importation
## ================

def load_base_dataset(x_path,y_path):
    '''
    Loads the basic signs set along with the labels and concatenates the two
    Parameters
    ==========
    x_path: string
        Path of the signs csv dataset (2304 grey scale values for the 12000+ instances).
    y_path: string
        Path of the labels of the signs loaded under x_path.

    Returns
    =======
    signs: Pandas.Dataframe
        Signs dataframe composed of lines of: pixel0, pixel1, ... pixel2303, label.
    signs_rd: Pandas.Dataframe
        Signs dataframe with rows randomized.
    '''
    # Import images vector values
    signs = pd.read_csv(x_path, sep=',',na_values='None')
    signs.name = "Signs"
    # Import labels
    labels = pd.read_csv(y_path, sep=',',na_values='None')
    # Link label to vectors
    signs.insert(0,"label",labels.values)
    # Line Randomisation
    signs_rd = randomise(signs)
    return signs, signs_rd

# One label only data sets (boolean mask)
# =======================================

def load_dataset_with_boolean_mask(nb):
    '''
    Load the provided boolean mask y_train_smpl_<nb> linked to the base dataset.
    All labels are 1 except <nb> that is 0.
    Parameters
    ==========
    nb: int
        Label to be loaded.

    Returns
    =======
    signs_1label: Pandas.Dataframe
        Signs dataset along with a boolean mask for <nb>.
    '''
    global path_x_train
    path_labels  = path_boolean_mask(nb)
    signs_1label = pd.read_csv(path_x_train, sep=',',na_values='None')
    labels = pd.read_csv(path_boolean_mask(nb), sep=',',na_values='None')
    signs_1label.insert(0,"label",labels.values)
    signs_1label.name = "Signs with boolean mask for label" + str(nb)
    return signs_1label

## Data visualisation
## ==================

def print_head_tail(df):
    '''
    Prints the first and last n elements of the dataframe
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataframe to be visualised.
    '''
    print("Head:\n{0}".format(df.head(n=5)))
    print("Tail:\n{0}".format(df.tail(n=5)))

def plot_feature(df,feature):
    '''
    Plot the number of the feature attribute in the given dataset.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataframe from which the feature is extracted.
    feature: string
        Feature to be extracted and shown.
    '''
    print(df.groupby(feature).size())
    plt.figure()
    sns.set(style="whitegrid", color_codes=True)
    sns.countplot(x='label',data=df)
    plt.show()

def display_nth_sign(df,n,feature='label'):
    '''
    Display the nth sign of the dataset.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset from which the element will be taken.
    n: int
        Row of the dataset.
    '''
    cols = [col for col in df.columns if col!=feature]
    plt.figure()
    plt.axis('off')
    plt.imshow(df[cols].loc[n].values.reshape((48,48)),cmap='Greys')
    plt.show()


## Filters
## =======

def randomise(df):
    '''
    Randomise the order of the instances.
    Parameters
    ==========
    df: Pandas.DataFrame
        Dataframe to randomise.

    Returns
    =======
    rd.df: Pandas.DataFrame
        Randomised dataframe.
    '''
    df_rd = df.sample(frac=1).reset_index(drop=True)
    df_rd.name = df.name + " randomised"
    return df_rd

def select_instances(df,feature,rd=True):
    '''
    Select, based on the feature class, the smallest number of instances
    that will be taken from each of the different feature classes and output
    the dataframe produced this way.
    Parameters
    ==========
    df: Pandas.DataFrame
        Dataframe that needs to be reduced.
    feature: string
        Feature the reducing will be based upon.

    Returns
    =======
    reduced_df: Pandas.Dataframe
        Portion of the base dataframe with the same number of instances
        selected for each feature class.
    '''
    df_grouplab = df.groupby(feature)
    min_instances = min(df_grouplab.size())
    if rd:
        sm_df = pd.DataFrame(df_grouplab.head(n=min_instances).reset_index(drop=True))
        sm_df.name = "reduced Dataframe '{0}'".format(df.name)
        return sm_df
    else:
        sm_df = pd.DataFrame(df_grouplab.sample(n=min_instances).reset_index(drop=True))
        sm_df.name = "reduced Dataframe (random selection) '{0}'".format(df.name)
        return sm_df

## Transformations
## ===============

def normalise_dataset(df,class_feature):
    '''
    Normalise the dataset by using the preprocessing functions coming in sklearn.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataframe to be normalised.
    class_feature: string
        Feature that has NOT to be taken in consideration by the normalisation.
    '''
    fc = df[class_feature]
    cols = [col for col in df.columns if col!=class_feature]
    dataset_without_fc = df[cols]
    normalised_df = pd.DataFrame(preprocessing.scale(dataset_without_fc))
    normalised_df.insert(0,"label",fc.values)
    normalised_df.name = df.name + ' normalised'
    return normalised_df

def divide_by_255(df,class_feature):
    '''
    Normalise the dataset by dividing the grey scale pixels by 255.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataframe to be normalised.
    class_feature: string
        Feature that has NOT to be taken in consideration by the normalisation.
    '''
    fc = df[class_feature]
    cols = [col for col in df.columns if col!=class_feature]
    dataset_without_fc = df[cols]
    normalised_df = dataset_without_fc.astype('float')/255
    normalised_df.insert(0,"label",fc.values)
    normalised_df.name = df.name + ' /255'
    return normalised_df

## Prepare training/test data
## ==========================

def separate_train_test(df,class_feature, ratio=0.20):
    '''
    Extract the class feature from the dataset and creates a train/test dataset
    as well as the corresponding train/test targets.
    Parameters
    ==========
    df: Pandas.Dataframe
        The dataframe that needs to be split.
    class_feature: string
        Name of the class feature (column of the dataframe).
    ratio: float
        Ration of train/test that needs to be done. Default: 0.2 (0.8 train/0.2 test).
    '''
    cols   = [col for col in df.columns if col!=class_feature]
    data   = df[cols]
    target = df[class_feature]

    # Separation between training/test data with the given ratio
    data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = ratio, random_state = 10)
    return data_train, data_test, target_train, target_test


if __name__ == "__main__":
    # # Loading base dataset and sampling/randomising it
    signs, signs_rd = load_base_dataset(path_x_train,path_y_train)
    sm_signs = select_instances(signs,'label')
    sm_signs_rd = randomise(sm_signs)

    # Display the instances sorted by labels
    plot_feature(signs,'label')
    plot_feature(sm_signs,'label')

    signs_rd_norm1 = divide_by_255(signs_rd,'label')
    signs_rd_norm2 = normalise_dataset(signs_rd,'label')
    sm_signs_rd_norm1 = divide_by_255(sm_signs_rd,'label')
    sm_signs_rd_norm2 = normalise_dataset(sm_signs_rd,'label')

    # Print the beginning/end of the datasets
    # print_head_tail(signs_rd)
    # print_head_tail(sm_signs_rd)
    # print_head_tail(sm_signs_rd_norm1)
    # print_head_tail(sm_signs_rd_norm2)

    display_nth_sign(signs,1300)
    display_nth_sign(signs,5600)
    display_nth_sign(signs,10400)
    display_nth_sign(signs,12650)
