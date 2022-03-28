import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


INPUT_DIR = os.path.join(os.getcwd(), 'data', 'processed')
OUTPUT_DIR = INPUT_DIR
VOCAB_SIZE = 8192
ENCODING = 'utf-8'


def main():
    # read input
    fname = os.path.join(INPUT_DIR, 'manifestos.csv')
    df_texts = pd.read_csv(fname)
    fname = os.path.join(INPUT_DIR, 'clusters.csv')
    df_clusters = pd.read_csv(fname)

    # vectorize texts and split data
    vectorizer = TfidfVectorizer(encoding=ENCODING, max_features=VOCAB_SIZE)
    X = vectorizer.fit_transform(df_texts['text'])
    y = np.array(df_clusters['cluster'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.1, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train)

    # export arrays
    names = ('train', 'val', 'test')
    for x, name in zip((X_train, X_val, X_test), names):
        np.savetxt(
            os.path.join(OUTPUT_DIR, f'tf-idf_X{name}.txt'),
            x.toarray()
        )
    for y, name in zip((y_train, y_val, y_test), names):
        np.savetxt(os.path.join(OUTPUT_DIR, f'tf-idf_y{name}.txt'), y)


if __name__ == '__main__':
    main()
