'''
provide image transform functions for mnist-like datasets.
'''
import numpy as np


def mnist_to_4x4(img_vec: np.ndarray):
    '''
    transform mnist image to 4x4 image.
    '''
    img = img_vec.reshape((28,28)).astype(np.float32)
    img *= 1.0/255.0
    img = (img-0.1307) / 0.3081
    img = img[2:26, 2:26]
    # avgpooling
    M, N = 24,24
    K = 6
    L = 6
    MK = M // K
    NL = N // L
    img = img[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))
    return img.flatten()


def mnist_to_28x28(img_vec: np.ndarray):
    '''
    transform mnist image to 28x28 image.
    '''
    img = img_vec.reshape((28,28)).astype(np.float32)
    img *= 1.0/255.0
    img = (img-0.1307) / 0.3081
    return img.flatten()

