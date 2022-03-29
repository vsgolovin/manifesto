import os
import numpy as np
from sklearn import svm


DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')
PREFIX = 'tf-idf_'


def main():
    # load data
    X_train = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'Xtrain.txt'))
    y_train = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'ytrain.txt'),
                         dtype='uint8')
    X_test = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'Xtest.txt'))
    y_test = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'ytest.txt'),
                        dtype='uint8')

    model = svm.LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(list(zip(y_pred, y_test)))
    print((y_pred == y_test).sum() / len(y_test))


if __name__ == '__main__':
    main()
