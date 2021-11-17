import os

import fasttext

from .json_to_train import json_to_train
from .config import model_file_path, train_file_path


def train():
    print('Training...')
    json_to_train()
    model = fasttext.train_supervised(train_file_path, lr=0.25, epoch=30, wordNgrams=3)
    print('Compressing...')
    # model.quantize(input=train_file_path, retrain=True)
    print('Saving...')
    model.save_model(model_file_path)
    print('')
    print(f'Model saved to {model_file_path}.')

    labels = [label.replace('__label__', '') for label in model.labels]
    print('')
    print(f'LABELS ({len(labels)}):')
    for label in labels:
        print(f'- {label}')

    return model


def load_model():
    if os.path.exists(model_file_path):
        print('Model exists. Loading...')
        model = fasttext.load_model(model_file_path)
        print(f'Loaded {model_file_path}')
    else:
        print('Model does not exist. Training...')
        model = train()

    return model


if __name__ == '__main__':
    train()
