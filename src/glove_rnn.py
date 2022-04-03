from typing import Callable
import os
from collections import Counter
from functools import partial
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchtext.vocab as tvcb
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt


DATA_DIR = os.path.join(os.getcwd(), 'data')
BATCH_SIZE = 10
TEST_SIZE = 40
LEARNING_RATE = 5e-4
EPOCHS = 50
TOKENIZER = get_tokenizer('spacy', 'en_core_web_sm')
EMBEDDINGS = tvcb.GloVe('6B', dim=50)
OUTPUT_PATH = os.path.join(os.getcwd(), 'reports')


def main():
    # load and split data
    df_texts = pd.read_csv(
        os.path.join(DATA_DIR, 'interim', 'all_manifestos.csv'),
        encoding='utf-8'
    )
    df_clusters = pd.read_csv(os.path.join(DATA_DIR, 'processed',
                                           'clusters.csv'))
    NUM_CLASSES = len(df_clusters['cluster'].unique())
    X_train, X_test, y_train, y_test = train_test_split(
        df_texts['text'], df_clusters['cluster'],
        test_size=TEST_SIZE,
        stratify=df_clusters['cluster'],
        shuffle=True
    )

    # count words in corpus
    counter = Counter('<pad>')
    for text in df_texts['text']:
        counter.update(TOKENIZER(text))
    myvocab = tvcb.vocab(counter)    # str -> int
    myvec = EMBEDDINGS.get_vecs_by_tokens(  # int -> Tensor
        myvocab.get_itos(), lower_case_backup=True)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # create `DataLoader`s for iterating over data
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    collate_fn = partial(
        collate_batch,
        text_pipeline=lambda s: myvocab(TOKENIZER(s)),
        device=device
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    model = RNN(myvec, myvec.size(1), NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.NLLLoss()
    train_losses = np.zeros(EPOCHS)
    train_accuracies = np.zeros_like(train_losses)
    val_losses = np.zeros_like(train_losses)
    val_accuracies = np.zeros_like(train_losses)

    for epoch in range(EPOCHS):
        print(f'[Epoch #{epoch + 1}]')
        loss, acc = train(train_dataloader, model, loss_function, optimizer)
        print(f'Train: loss = {loss:.2e}, accuracy = {acc:3f}')
        train_losses[epoch] = loss
        train_accuracies[epoch] = acc
        loss, acc = evaluate(test_dataloader, model, loss_function)
        print(f'Validate: loss = {loss:.2e}, accuracy = {acc:3f}\n')
        val_losses[epoch] = loss
        val_accuracies[epoch] = acc

    epochs = np.arange(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='validate')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'rnn_loss.png'), dpi=150)

    plt.figure()
    plt.plot(epochs, train_accuracies, label='train')
    plt.plot(epochs, val_accuracies, label='validate')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'rnn_accuracy.png'), dpi=150)


class TextDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series):
        self.texts = list(texts)
        self.labels = np.array(labels, dtype='uint8')

    def __getitem__(self, i: int) -> tuple[str, int]:
        return self.texts[i], self.labels[i]

    def __len__(self):
        return len(self.texts)


def collate_batch(batch: tuple, text_pipeline: Callable, device: torch.device):
    text_list, label_list = [], []
    for (_text, _label) in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(_label)
    text_tensor = pad_sequence(text_list, padding_value=0)
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    return text_tensor.to(device), label_tensor.to(device)


class RNN(nn.Module):
    def __init__(self, embeddings, hidden_size, num_classes):
        super().__init__()
        self.input_size = embeddings.size(1)
        self.hidden_size = hidden_size
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=hidden_size)
        self.clf = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X: torch.Tensor):
        output = self.emb(X)
        output = self.rnn(output)[0][-1]
        output = self.clf(output)
        return output


def train(dataloader: DataLoader, model: nn.Module, loss_function, optimizer):
    total_loss = 0.0
    accurate = 0
    total_samples = 0
    model.train()
    for X, y in dataloader:
        output = model.forward(X)
        loss = loss_function(output, y)
        total_samples += y.size(0)
        total_loss += loss.detach().cpu().item() * y.size(0)
        pred = torch.argmax(output.detach().cpu(), dim=1)
        accurate += (pred == y.cpu()).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / total_samples, accurate / total_samples


def evaluate(dataloader: DataLoader, model: nn.Module, loss_function):
    total_loss = 0.0
    accurate = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            output = model.forward(X)
            pred = torch.argmax(output.cpu(), axis=1)
            accurate += (pred == y.cpu()).sum()
            total_samples += y.size(0)
            loss = loss_function(output, y)
            total_loss += loss.cpu().item() * y.size(0)
    return total_loss / total_samples, accurate / total_samples


if __name__ == '__main__':
    main()
