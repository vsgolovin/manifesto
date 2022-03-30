import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


DATA_DIR = os.path.join(os.getcwd(), 'data')
BATCH_SIZE = 16
TEST_SIZE = 40
GLOVE_NAME = '6B'
GLOVE_DIM = 50


def main():
    # load and split data
    df_texts = pd.read_csv(
        os.path.join(DATA_DIR, 'interim', 'all_manifestos.csv'),
        encoding='utf-8'
    )
    df_clusters = pd.read_csv(os.path.join(DATA_DIR, 'processed',
                                           'clusters.csv'))
    X_train, X_test, y_train, y_test = train_test_split(
        df_texts['text'], df_clusters['cluster'],
        test_size=TEST_SIZE,
        stratify=df_clusters['cluster'],
        shuffle=True
    )

    # create `DataLoader`s for iterating over data
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_SIZE)

    for batch in train_dataloader:
        print(type(batch[0]))


class TextDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series):
        self.texts = list(texts)
        self.labels = np.array(labels, dtype='uint8')

    def __getitem__(self, i: int) -> tuple[str, int]:
        return self.texts[i], self.labels[i]

    def __len__(self):
        return len(self.texts)


if __name__ == '__main__':
    main()
