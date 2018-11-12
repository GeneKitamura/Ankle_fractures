import numpy as np
import pandas as pd

def get_root_data(batch_sz, val_test_multiples, x_loc, y_loc):
    # validation and test set sizes are multiples of the BATCH_SZ

    X = np.load(x_loc)
    Y = np.load(y_loc)

    end_train = int(2 * batch_sz * val_test_multiples) # end of training sample
    end_val = int(batch_sz * val_test_multiples) # end of validation sample
    x_train = X[:-end_train]
    y_train = Y[:-end_train]

    # Test and validation sets are equal in size
    x_val = X[-end_train:-end_val]
    y_val = Y[-end_train:-end_val]
    x_test = X[-end_val:]
    y_test = Y[-end_val:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def _strict_data(batch_sz, val_test_multiple, x_loc, y_loc, return_triples=False):
    x_train = np.load('./data/reshaped_single_views.npy')
    y_train = np.load('./data/single_views_y.npy')
    x_val = np.load('./data/val_single_views.npy')
    y_val = np.load('./data/val_single_y.npy')
    x_test = np.load('./data/test_single_views.npy')
    y_test = np.load('./data/test_single_y.npy')
    triple_val_x = np.load('./data/val_triple_views.npy')
    triple_val_y = np.load('./data/val_triple_y.npy')
    triple_test_x = np.load('./data/test_triple_views.npy')
    triple_test_y = np.load('./data/test_triple_y.npy')

    if return_triples:
        return (triple_val_x, triple_val_y, triple_test_x, triple_test_y)
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test

def dummy_data(batch_sz, val_test_multiples, x_loc, y_loc):
    x_train = np.random.randn(batch_sz*20, 299, 299, 1)
    y_train = np.random.randint(2, size=batch_sz*20)
    x_val = np.random.randn(batch_sz*val_test_multiples, 299, 299, 1)
    y_val = np.random.randint(2, size=batch_sz*val_test_multiples)
    x_test = np.random.randn(batch_sz*val_test_multiples, 299, 299, 1)
    y_test = np.random.randint(2, size=batch_sz*val_test_multiples)

    return x_train, y_train, x_val, y_val, x_test, y_test

def mnist_data(batch_sz, val_test_multiples, x_loc, y_loc):
    train_set = pd.read_csv('./mnist/mnist_train.csv', header=None)
    test_set = pd.read_csv('./mnist/mnist_test.csv', header=None)

    # get labels in own array
    train_lb = np.array(train_set[0])
    test_lb = np.array(test_set[0])

    # drop the labels column from training dataframe
    trainX = train_set.drop(0, axis=1)
    testX = test_set.drop(0, axis=1)

    # put in correct float32 array format
    trainX = np.array(trainX).astype(np.float32)
    testX = np.array(testX).astype(np.float32)

    # reformat the data so it's not flat
    trainX = trainX.reshape(len(trainX), 28, 28, 1)
    testX = testX.reshape(len(testX), 28, 28, 1)

    # get a validation set and remove it from the train set
    trainX_m = trainX[0:(len(trainX) - 500), :, :, :]
    valX = trainX[(len(trainX) - 500):len(trainX), :, :, :]
    train_lb_m = train_lb[0:(len(trainX) - 500)]
    val_lb = train_lb[(len(trainX) - 500):len(trainX)]

    return trainX_m, train_lb_m, valX, val_lb, testX, test_lb

def overfitting_data(batch_sz, val_test_multiples, x_loc, y_loc):
    x_train = np.load('./data/val_single_views.npy')
    y_train = np.load('./data/val_single_y.npy')
    x_val = np.load('./data/val_single_views.npy')
    y_val = np.load('./data/val_single_y.npy')
    x_test = np.load('./data/val_single_views.npy')
    y_test = np.load('./data/val_single_y.npy')

    return x_train, y_train, x_val, y_val, x_test, y_test



