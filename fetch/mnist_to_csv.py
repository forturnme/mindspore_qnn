'''
implement the function to convert mnist dataset to csv file.
this script finishes these tasks:
first, download the MNIST dataset to data/raw/mnist/figure/
if the directory does not exist, create it.
second, convert the dataset to csv file and save it to data/raw/mnist/csv/.
the filename is mnist_train.csv and mnist_test.csv.
third, convert the dataset to 4x4 image vectors and select the first n classes and save it 
to data/mnist4x4/. the filenames will be mnist4x4_n_train_labels.csv, 
mnist4x4_n_train_images.csv, mnist4x4_n_test_labels.csv and mnist4x4_n_test_images.csv.
n is between 2 and 10.
'''
import os
import gzip
import numpy as np
from transformations import mnist_to_4x4


def download_mnist(img_path):
    '''
    download mnist dataset to `img_path`.
    '''
    import urllib.request
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    filenames = ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']
    for filename in filenames:
        print('downloading '+filename+'...')
        urllib.request.urlretrieve(base_url+filename, img_path+filename)
        print('done.')


def load_mnist(path, kind='train'):
    '''
    load mnist dataset from `path`.
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


def mnist_to_csv(img_path, csv_path, kind='train'):
    '''
    convert mnist dataset to csv file.
    '''
    images, labels = load_mnist(img_path, kind)
    data = np.concatenate((labels.reshape((-1,1)), images), axis=1)
    if kind == 't10k':
        kind = 'test'
    np.savetxt(csv_path+'mnist_'+kind+'.csv', data, delimiter=',')
    return data


def mnist_to_4x4_classes_csv(img_path, csv_path, n_classes, kind='train'):
    '''
    convert mnist dataset to 4x4 image vectors and select the first n classes and save it 
    to data/mnist4x4/. the filenames will be mnist4x4_n_train_labels.csv, 
    mnist4x4_n_train_images.csv, mnist4x4_n_test_labels.csv and mnist4x4_n_test_images.csv.
    n is between 2 and 10.
    '''
    images, labels = load_mnist(img_path, kind)
    data = np.concatenate((labels.reshape((-1,1)), images), axis=1)
    idx = data[:,0]<n_classes
    data = data[idx, :]
    if kind == 't10k':
        kind = 'test'
    np.savetxt(csv_path+'mnist4x4_'+str(n_classes)+'_'+kind+'_labels.csv', data[:,0], delimiter=',', fmt='%f')
    data = np.array([mnist_to_4x4(img) for img in data[:,1:]])
    np.savetxt(csv_path+'mnist4x4_'+str(n_classes)+'_'+kind+'_images.csv', data, delimiter=',', fmt='%f')
    return data


if __name__ == '__main__':
    img_path = 'data/raw/mnist/figure/'
    csv_path = 'data/raw/mnist/csv/'
    d4x4_path = 'data/mnist4x4/'
    if not os.path.exists(d4x4_path):
        os.makedirs(d4x4_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    # if dataset does not exist, download it.
    if not os.path.exists(img_path+'train-images-idx3-ubyte.gz'):
        download_mnist(img_path)
    mnist_to_csv(img_path, csv_path, 'train')
    mnist_to_csv(img_path, csv_path, 't10k')
    # convert mnist to 4x4 image vectors and select the first n classes.
    for n_classes in range(2,11):
        print('now converting mnist to 4x4 image vectors and select the first '+str(n_classes)+' classes.')
        mnist_to_4x4_classes_csv(img_path, d4x4_path, n_classes, 'train')
        mnist_to_4x4_classes_csv(img_path, d4x4_path, n_classes, 't10k')
    print('done.')

