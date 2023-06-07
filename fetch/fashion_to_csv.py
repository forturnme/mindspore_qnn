'''
implement the function to convert fashion-mnist dataset to csv file.
this script finishes these tasks:
first, download the fashion-MNIST dataset to data/raw/fashion/figure/
if the directory does not exist, create it.
second, convert the dataset to csv file and save it to data/raw/fashion/csv/.
the filename is fashion_train.csv and fashion_test.csv.
third, convert the dataset to 4x4 image vectors and select the first n classes and save it 
to data/fashion4x4/. the filenames will be fashion4x4_n_train_labels.csv, 
fashion4x4_n_train_images.csv, fashion4x4_n_test_labels.csv and fashion4x4_n_test_images.csv.
n is between 2 and 10.
'''
import os
import gzip
import numpy as np
from transformations import mnist_to_4x4


def download_fashion(img_path):
    '''
    download fashion-mnist dataset to `img_path`.
    '''
    import urllib.request
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    filenames = ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']
    for filename in filenames:
        print('downloading '+filename+'...')
        urllib.request.urlretrieve(base_url+filename, img_path+filename)
        print('done.')

def load_fashion(path, kind='train'):
    '''
    load fashion-mnist dataset from `path`.
    '''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    
    return images, labels


def fashion_to_csv(img_path, csv_path, kind='train'):
    '''
    convert fashion-mnist dataset to csv file.
    '''
    images, labels = load_fashion(img_path, kind)
    data = np.concatenate((labels.reshape((-1,1)), images), axis=1)
    if kind == 't10k':
        kind = 'test'
    np.savetxt(csv_path+'fashion_'+kind+'.csv', data, delimiter=',', fmt='%d')


def fashion_to_4x4(img_path, csv_path, kind='train', classes=10):
    '''
    convert fashion-mnist dataset to 4x4 image vectors and select the first n classes.
    '''
    images, labels = load_fashion(img_path, kind)
    data = np.concatenate((labels.reshape((-1,1)), images), axis=1)
    if kind == 't10k':
        kind = 'test'
    np.savetxt(csv_path+'fashion_'+kind+'.csv', data, delimiter=',', fmt='%d')
    # convert to 4x4 image vectors
    data = np.concatenate((labels.reshape((-1,1)), np.apply_along_axis(mnist_to_4x4, 1, images)), axis=1)
    np.savetxt(csv_path+'fashion4x4_'+str(classes)+'_'+kind+'_labels.csv', data[:,0], delimiter=',', fmt='%f')
    np.savetxt(csv_path+'fashion4x4_'+str(classes)+'_'+kind+'_images.csv', data[:,1:], delimiter=',', fmt='%f')


if __name__ == '__main__':
    # download mnist dataset
    img_path = 'data/raw/fashion/figure/'
    csv_path = 'data/raw/fashion/csv/'
    d4x4_path = 'data/fashion4x4/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(d4x4_path):
        os.makedirs(d4x4_path)
    if not os.path.exists(img_path+'train-images-idx3-ubyte.gz'):
        download_fashion(img_path)
    # convert mnist dataset to csv file
    fashion_to_csv(img_path, csv_path, kind='train')
    fashion_to_csv(img_path, csv_path, kind='t10k')
    # convert mnist dataset to 4x4 image vectors
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for nc in range(2,11):
        print('converting fashion-mnist to 4x4 image vectors with %d classes...' % nc)
        fashion_to_4x4(img_path, d4x4_path, kind='train', classes=nc)
        fashion_to_4x4(img_path, d4x4_path, kind='t10k', classes=nc)
    print('done.')
    
