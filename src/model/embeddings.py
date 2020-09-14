from typing import List, Tuple
import math as m

import tensorflow.keras.preprocessing.text as t

import src.model.models.lstm_autoencoder as la
import src.model.data as d
import src.card_data as c

BATCH_SIZE = 128
LSTM_UNITS = 128
EPOCHS = 4
EMBEDDING_LEN = 8
RaggedIntArr = List[List[int]]


def get_longest_sent_size(texts: List[List[str]]) -> int:
    return max(map(len, texts))


def vocab_size(texts: List[List[str]]) -> int:
    words = set()
    for sent in texts:
        words.update(sent)
    return len(words)


def get_tokenizer(text: List[List[str]]) -> Tuple[RaggedIntArr, t.Tokenizer]:
    tokenizer = t.Tokenizer()
    tokenizer.fit_on_texts(text)
    tokens = tokenizer.texts_to_sequences(text)
    return tokens, tokenizer


def optimal_embedding_len(vocab_size: int) -> float:
    return m.ceil(vocab_size ** .25)


if __name__ == '__main__':
    texts = c.load_pickle(c.CARD_TEXTS_PATH)
    max_sent_len = get_longest_sent_size(texts)
    unique_word_count = vocab_size(texts)
    encoded_texts, tokenizer = get_tokenizer(texts)

    x_train, y_train, x_test = d.split_training_data(encoded_texts)
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
