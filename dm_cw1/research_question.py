'''
# created 01/11/2019 11:21
# by Q.Ducasse
'''

import cv2
import sklearn
from PIL import Image
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt

def load_test_image():
    '''
    Load the base image on which we will perform the different tests/filters.
    '''
    image = cv2.imread('./data/test_image.jpg')
    return image

def mean_filter(image,ksize=3):
    '''
    Apply a basic mean filter on the test image.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset that will pass through the filter.

    Returns
    =======
    filt_df
    '''
    filt_im = cv2.blur(image,(ksize,ksize))
    return filt_im

def gaussian_filter(image,ksize=3,mean=0):
    '''
    Apply a gaussian filter on the test image. Similar to the mean filter while
    adding the weighted average of the surrounding pixels.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset that will pass through the filter.

    Returns
    =======
    filt_df
    '''
    filt_im = cv2.GaussianBlur(image,(ksize,ksize),mean)
    return filt_im

def median_filter(image,ksize=3):
    '''
    Apply a median filter on the test image. Calculate the median of the pixel
    intensities surrounding a given pixel.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset that will pass through the filter.

    Returns
    =======
    filt_df
    '''
    filt_im = cv2.medianBlur(image,ksize)
    return filt_im

def laplacian_filter(image,ksize=3):
    '''
    Apply a laplacian filter on the test image. Similar to the mean filter while
    adding the weighted average of the surrounding pixels.
    Parameters
    ==========
    df: Pandas.Dataframe
        Dataset that will pass through the filter.

    Returns
    =======
    filt_df
    '''
    filt_im = cv2.Laplacian(image,cv2.CV_64F)
    return filt_im

def display_image(base_im,im,title):
    '''
    Shows the image passed in argument along with the base image.
    Parameters
    ==========
    base_im: array
        Base test image.
    im: array
        Filtered image.
    title: string
        Title of the second image.
    '''
    plt.figure()
    plt.subplot(121)
    plt.imshow(base_im)
    plt.title('Initial Image')
    plt.subplot(122)
    plt.imshow(im)
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    base_im = load_test_image()

    im_mean = mean_filter(base_im)
    display_image(base_im,im_mean,'Mean Filter applied')

    im_gauss = gaussian_filter(base_im)
    display_image(base_im,im_gauss,'Gaussian Filter applied')

    im_med = median_filter(base_im)
    display_image(base_im,im_med,'Median Filter applied')

    im_lap = laplacian_filter(base_im)
    display_image(base_im,im_lap,'Laplacian Filter applied')
