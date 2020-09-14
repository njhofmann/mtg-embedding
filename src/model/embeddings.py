import pathlib as pl
import pickle as p
from typing import List, Tuple

import numpy as np
import tensorflow.keras.preprocessing.text as t

import src.model.lstm_autoencoder as la
import src.model.data as d

BATCH_SIZE = 128
LSTM_UNITS = 128
EPOCHS = 4
EMBEDDING_LEN = 8
RaggedIntArr = List[List[int]]
PICKLE_DIRC = pl.Path(__file__).parent.parent.parent.joinpath('pickle-files/card_texts.pickle')


def load_data() -> Tuple[List[List[str]], int, int]:
    with open(PICKLE_DIRC, 'rb') as f:
        texts = p.load(f)
    unique_texts = list(map(list, set(map(tuple, texts))))
    num_of_texts = len(texts)
    text_len = max(map(len, texts))
    return unique_texts, num_of_texts, text_len


def get_tokenizer(text: List[List[str]]) -> Tuple[RaggedIntArr, t.Tokenizer]:
    tokenizer = t.Tokenizer()
    tokenizer.fit_on_texts(text)
    tokens = tokenizer.texts_to_sequences(text)
    return tokens, tokenizer


if __name__ == '__main__':
    texts, text_count, max_sent_len = load_data()
    unique_word_count = len(np.unique(texts))
    encoded_text, tokenizer = get_tokenizer(texts)

    x_train, y_train, x_test = d.split_training_data(encoded_text)
    x_train = d.convert_to_tensor(x_train, max_sent_len)
    y_train = d.convert_to_tensor(y_train, max_sent_len)
    x_test = d.convert_to_tensor(x_test, max_sent_len)

    autoencoder = la.create_seq_2_seq_autoencoder(embedding_len=EMBEDDING_LEN,
                                                  word_count=unique_word_count,
                                                  sent_len=max_sent_len,
                                                  lstm_units=LSTM_UNITS)
    print(autoencoder.summary())
    autoencoder = la.train_autoencoder(autoencoder, x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    print(autoencoder.evaluate(x=x_test, y=x_test))
