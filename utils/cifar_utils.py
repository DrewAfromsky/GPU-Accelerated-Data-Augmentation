# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

################################################################
# Code adapted from ECBM 4040, @Columbia University, Fall 2019 #
################################################################

# This is a utility function to download the CIFAR dataset and preprocess the data.

# Import modules
# import _pickle as pickle
import pickle
import os
import tarfile
import glob
import urllib.request as url
import numpy as np
import matplotlib.pyplot as plt


def download_data():
    
    """
    Download the CIFAR-10 data from the website (~ 170MB).
    The data (.tar.gz file) will be store in the ./data/ folder.
    :return: None
    """

    if not os.path.exists('./data'):
        os.mkdir('./data')
        print('Start downloading data...')
        url.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                        "./data/cifar-10-python.tar.gz")
        print('Download complete.')
    else:
        if os.path.exists('./data/cifar-10-python.tar.gz'):
            print('CIFAR-10 package already exists.')


def load_data(mode='all'):

    """
    Unpack the CIFAR-10 dataset and load the datasets.
    :param mode: 'train', or 'test', or 'all'. Specify the training set or test set, or load all the data.
    :return: A tuple of data/labels, depending on the chosen mode. If 'train', return training data and labels;
    If 'test' ,return test data and labels; If 'all', return both training and test sets.
    """
    
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists('./data/cifar-10-python.tar.gz'):
        download_data()
    else:
        print('./data/cifar-10-python.tar.gz already exists. Begin extracting...')
    # Check if the package has been unpacked, otherwise unpack the package
    if not os.path.exists('./data/cifar-10-batches-py/'):
        package = tarfile.open('./data/cifar-10-python.tar.gz')
        package.extractall('./data')
        package.close()
    # Go to the location where the files are unpacked
    root_dir = os.getcwd()
    os.chdir('./data/cifar-10-batches-py')
    
    images = []
    data_train = glob.glob('data_batch*')

    try:
        for name in data_train:
            handle = open(name, 'rb')
            cmap = pickle.load(handle, encoding='bytes')
            images.append(cmap[b'data'])
            handle.close()
        # Turn the dataset into numpy compatible arrays.
        images = np.concatenate(images, axis=0)
    except BaseException:
        print('Something went wrong...')
        return None

    os.chdir(root_dir)
    
    if mode == 'all':
        return images
    else:
        raise ValueError('Mode should be \'all\'')

def show(images):

    """
    Plot the top 16 images (index 0~15) for visualization.
    :param images: images to be shown
    """

    xshow = images[:25]
    fig = plt.figure(figsize=(6,6))
    fig.set_tight_layout(True)
    
    for i in range(25):
        ax = fig.add_subplot(5,5,i+1)
        # ax.imshow((xshow[i,:]*255).astype(np.uint8))
        ax.imshow((xshow[i,:]))

        ax.axis('off')

