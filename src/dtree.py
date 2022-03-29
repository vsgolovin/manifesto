import os
import numpy as np
from sklearn import tree


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

    model = tree.DecisionTreeClassifier(min_samples_leaf=5, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).sum() / len(y_test)
    print(list(zip(y_pred, y_test)))
    print(acc)


if __name__ == '__main__':
    main()
