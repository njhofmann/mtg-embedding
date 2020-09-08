import numpy as np
import pickle as p
from typing import List, Tuple
import random as r
import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow.keras.models as m
import tensorflow.keras.preprocessing.sequence as s
import tensorflow.keras.preprocessing.text as t

PADDING_SYMBOL = 0
BATCH_SIZE = 64


def load_data() -> Tuple[List[List[str]], int, int]:
    with open('pickle-files/card_texts.pickle', 'rb') as f:
        texts = p.load(f)
    # texts = np.load('numpy-files/card_texts.npy', allow_pickle=True)
    unique_texts = list(map(list, set(map(tuple, texts))))
    num_of_texts = len(texts)
    text_len = max(map(len, texts))
    return unique_texts, num_of_texts, text_len


def get_tokenizer(text: List[List[str]]) -> Tuple[List[List[int]], t.Tokenizer]:
    tokenizer = t.Tokenizer()
    tokenizer.fit_on_texts(text)
    tokens = tokenizer.texts_to_sequences(text)
    return tokens, tokenizer


def get_encoder(sent_len: int, embedding_len: int, word_count: int, lstm_units: int) -> Tuple[m.Model, l.Layer]:
    encoder_inputs = l.Input(shape=(sent_len,), name='encoder-input')
    embedded_inputs = l.Embedding(input_dim=word_count, output_dim=embedding_len, input_length=sent_len,
                                  name='encoder-embedding', mask_zero=False)(encoder_inputs)
    state_h = l.Bidirectional(l.LSTM(lstm_units, activation='relu', name='encoder-lstm'))(embedded_inputs)
    return m.Model(inputs=encoder_inputs, outputs=state_h, name='encoder-model')(encoder_inputs), encoder_inputs


def get_decoder(encoder_output: m.Model, sent_len: int, word_count: int, lstm_units: int) -> l.Layer:
    decoded = l.RepeatVector(n=sent_len)(encoder_output)  # TODO what does this do
    decoder_lstm = l.Bidirectional(l.LSTM(lstm_units, return_sequences=True, name='decoder-dense'))(decoded)
    return l.Dense(word_count, activation='softmax', name='decoder-dense')(decoder_lstm)


def get_seq_2_seq_autoencoder(embedding_len: int, word_count: int, sent_len: int, lstm_units: int) -> m.Model:
    """Creates a seq2seq autoencoder using the given parameters
    :param embedding_len: number of dimensions for the resulting embeddings wil
    :param word_count: the number of unique words in the input
    :param sent_len: how long each sentence is / length of the longest sequence
    :param lstm_units: number of LSTM cells to use in encoder and decoder
    :return: untrained seq2seq autoencoder
    """
    encoder, encoder_input = get_encoder(sent_len, embedding_len, word_count, lstm_units)
    decoder = get_decoder(encoder, sent_len, word_count, lstm_units)
    return m.Model(encoder_input, decoder)


def train_autoencoder(model: m.Model, data: np.ndarray, epochs: int, batch_size: int) -> m.Model:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=data, y=data, epochs=epochs, batch_size=batch_size)
    return model


def remove_rand_item(items: List[int], del_prob: float) -> List[int]:
    return [PADDING_SYMBOL if r.random() < del_prob else item for item in items]


def swap_rand_items(items: List[int]) -> List[int]:
    # TODO return None if less than size, stop when padding starts
    idxs = r.sample(population=list(range(len(items))))


def augment_data(data: List[List[int]], del_prob: float = .05, swap_count: int = 3) -> Tuple[List[List[int]], List[List[int]]]:
    x_data, y_data = [], []

    for item in data:
        x_data.append(item)
        y_data.append(item)

        if (masked_item := remove_rand_item(item, del_prob)) != item:
            x_data.append(masked_item)
            y_data.append(item)

        for i in range(swap_count):
            swapped_item = swap_rand_items(item)
            y_data.append(item)

    return x_data, y_data


def split_training_data(tokens: List[List[int]], split: float = .8) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    r.shuffle(x=tokens)
    split_idx = round(len(tokens) * split)
    train, test = tokens[:split_idx], tokens[split_idx:]
    train_x, train_y = augment_data(train)
    return train_x, train_y, test


if __name__ == '__main__':
    texts, text_count, text_len = load_data()
    unique_word_count = len(np.unique(texts))
    encoded_text, tokenizer = get_tokenizer(texts)

    tensor_text = tf.convert_to_tensor(encoded_text)
    input_length = len(encoded_text[0])
    padded_tokens = s.pad_sequences(tokens, padding='post')

    autoencoder = get_seq_2_seq_autoencoder(embedding_len=16,
                                            word_count=unique_word_count,
                                            sent_len=input_length,
                                            lstm_units=64)
    autoencoder = train_autoencoder(autoencoder, data=tensor_text, batch_size=32, epochs=100)
