'''
download the UCI vowel dataset, split them in train and test set and save it 
as .csv files in data/vowel/. labels and data are saved separately.
the train test split is approximately 6:1.
before saving, the data is normalized to N(0.1307, 0.3081)
select first n classes and save to csv files.
the name for files will be vowel_n_train_labels.csv, vowel_n_train_data.csv,
vowel_n_test_labels.csv and vowel_n_test_data.csv.
n should be between 2 and 11.
download path is https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data
download file will be saved as data/vowel/vowel.csv
'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def download_vowel(path):
    '''
    download the UCI vowel dataset from `path`.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+'vowel.csv'):
        os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data -P '+path)
        os.system('mv '+path+'vowel-context.data '+path+'vowel.csv')

def load_vowel(path):
    '''
    load the UCI vowel dataset from `path`.
    '''
    if not os.path.exists(path+'vowel.csv'):
        download_vowel(path)
    data = pd.read_csv(path+'vowel.csv', header=None, sep=' ')
    return data

def vowel_to_csv(path):
    '''
    convert the UCI vowel dataset to csv file.
    '''
    data = load_vowel(path)
    # transform the data
    # notice we don't need the first 3 columns
    # and the labels are in the last column
    # what else, the second last column is useless
    data = data.values
    labels = data[:,-1]
    data = data[:,3:-2]
    # normalize the data
    data = (data-0.1307) / 0.3081
    # select first n classes
    for n_classes in range(2,11):
        print('now selecting first '+str(n_classes)+' classes.')
        idx = labels<n_classes
        data_n = data[idx, :]
        labels_n = labels[idx]
        # split the data into train and test set
        data_train, data_test, labels_train, labels_test = train_test_split(data_n, labels_n, test_size=0.15, random_state=42)
        # save the data
        np.savetxt(path+'vowel_'+str(n_classes)+'_train_labels.csv', labels_train, delimiter=',', fmt='%f')
        np.savetxt(path+'vowel_'+str(n_classes)+'_train_data.csv', data_train, delimiter=',', fmt='%f')
        np.savetxt(path+'vowel_'+str(n_classes)+'_test_labels.csv', labels_test, delimiter=',', fmt='%f')
        np.savetxt(path+'vowel_'+str(n_classes)+'_test_data.csv', data_test, delimiter=',', fmt='%f')


if __name__ == '__main__':
    path = 'data/vowel/'
    if not os.path.exists(path):
        os.makedirs(path)
    vowel_to_csv(path)
    print('done.')

