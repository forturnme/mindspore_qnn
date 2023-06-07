import numpy as np
folder = 'data/mnist_csv/'


mnist_train_csv = np.loadtxt(folder+'mnist_train.csv', delimiter=',')
mnist_test_csv = np.loadtxt(folder+'mnist_test.csv', delimiter=',')



for nc in range(2,11):
    idx_train, idx_test = mnist_train_csv[:,0]<nc, mnist_test_csv[:,0]<nc
    # for c in vl:
    #     idx_train = idx_train || mnist_train_csv[:,0] == c
    #     idx_test = idx_test || mnist_test_csv[:,0] == c
    # train_n = mnist_train_csv[0, :]
    # for r in range(len(mnist_train_csv)):
    #     if mnist_train_csv[r,0] in vl:
    #         train_n = np.append(train_n, mnist_train_csv[r], axis=0)
    train_n = mnist_train_csv[idx_train, :]
    test_n = mnist_test_csv[idx_test, :]
    print(train_n.shape, test_n.shape)
    train_label = train_n[:,0]
    train_label = train_label.reshape((-1,1))
    train_image = []
    # image transform
    for r in range(train_n.shape[0]):
        img = train_n[r, 1:].reshape((28,28))
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
        img = img.flatten()
        train_image.append(img.tolist())
    train_imgs = np.array(train_image)
    # train_n = np.concatenate((train_label,train_imgs), axis=1)

    test_label = test_n[:,0]
    test_label = test_label.reshape((-1,1))
    test_image = []
    # image transform
    for r in range(test_n.shape[0]):
        img = test_n[r, 1:].reshape((28,28))
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
        img = img.flatten()
        test_image.append(img.tolist())
    test_imgs = np.array(test_image)
    test_n = np.concatenate((test_label,test_imgs), axis=1)
    np.savetxt(folder+'mnist4x4_'+str(nc)+'_train_labels.csv',train_label,delimiter=',')
    np.savetxt(folder+'mnist4x4_'+str(nc)+'_train_images.csv',train_imgs,delimiter=',')
    np.savetxt(folder+'mnist4x4_'+str(nc)+'_test_labels.csv',test_label,delimiter=',')
    np.savetxt(folder+'mnist4x4_'+str(nc)+'_test_images.csv',test_imgs,delimiter=',')
    print('finished', nc)
