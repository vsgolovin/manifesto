import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')
PREFIX = 'tf-idf_'
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
EPOCHS = 1500


def main():
    # load data
    X_train = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'Xtrain.txt'))
    y_train = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'ytrain.txt'),
                         dtype='uint8')
    train_dataset = Dataset(X_train, y_train, batch_size=BATCH_SIZE)
    TRAIN_SIZE = X_train.shape[0]
    NUM_CLASSES = len(np.unique(y_train))
    X_test = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'Xtest.txt'))
    y_test = np.loadtxt(os.path.join(DATA_DIR, PREFIX + 'ytest.txt'),
                        dtype='uint8')

    # choose device and initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Perceptron(
        num_features=X_train.shape[1],
        num_classes=NUM_CLASSES
    )
    model = model.to(device)

    # train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = np.zeros(EPOCHS)
    train_acc = np.zeros_like(train_losses)
    test_losses = np.zeros_like(train_losses)
    test_acc = np.zeros_like(train_losses)
    loss_function = nn.NLLLoss()
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.uint8, device=device)

    for i in range(EPOCHS):

        # train loop
        mean_loss = 0.0
        true_count = 0
        for X_j, y_j in train_dataset:
            X_j = torch.tensor(X_j, dtype=torch.float32, device=device)
            y_j = torch.tensor(y_j, dtype=torch.int64, device=device)
            y_pred = model.forward(X_j)
            loss = loss_function(y_pred, y_j)
            mean_loss += loss.item() * X_j.shape[0] / TRAIN_SIZE
            predictions = torch.argmax(y_pred, axis=1)
            true_count += (predictions == y_j).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_dataset.shuffle()
        train_losses[i] = mean_loss
        train_acc[i] = true_count / TRAIN_SIZE

        # test / validation
        with torch.no_grad():
            y_pred = model.forward(X_test)
            test_losses[i] = loss_function(y_pred, y_test).item()
            pred = torch.argmax(y_pred, axis=1)
            test_acc[i] = (pred == y_test).sum() / len(y_test)

        # display progress
        print('\rEpoch {:4d}, train loss = {:.3f}, test loss = {:.3f}'.format(
                i + 1, train_losses[i], test_losses[i]
              ), end='')

    print('\nComplete')

    print(list(zip(pred.cpu().numpy(), y_test.cpu().numpy())))

    # plot results
    epochs = np.arange(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, test_losses, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_acc, label='train')
    plt.plot(epochs, test_acc, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


class NeuralNet(nn.Module):
    """
    Neural net with a single hidden layer.
    """
    def __init__(self, num_features, num_classes, shape_hidden):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(num_features, shape_hidden[0]),
            nn.ReLU(),
            nn.Linear(shape_hidden[0], shape_hidden[1]),
            nn.ReLU(),
            nn.Linear(shape_hidden[1], num_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class Perceptron(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64):
        self.X = X
        self.y = y
        self.num_samples, self.num_features = X.shape
        self.batch_size = batch_size
        self.inds = np.arange(self.num_samples)

    def __len__(self):
        return self.num_samples

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def shuffle(self):
        self.inds = np.random.permutation(self.num_samples)

    def __iter__(self):
        if self.batch_size is None:
            return self.X[self.inds], self.y[self.inds]
        start = 0
        while start < len(self):
            stop = min(start + self.batch_size, self.num_samples)
            inds = self.inds[start:stop]
            yield self.X[inds], self.y[inds]
            start = stop


if __name__ == '__main__':
    main()
