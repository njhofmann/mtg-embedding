import numpy as np
from typing import List, Tuple
import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow.keras.models as m
import tensorflow.keras.preprocessing.text as t

# embedding encoder, fixed length
# attention / lstm encoder, variable length

BATCH_SIZE = 64


def load_data() -> Tuple[List[List[str]], int, int]:
    texts = np.load('numpy-files/card_texts.npy')
    unique_texts = np.vstack(list({tuple(row) for row in texts}))
    num_of_texts, text_length = unique_texts.shape
    return unique_texts.tolist(), num_of_texts, text_length


def get_tokenizer(text: List[List[str]]) -> Tuple[List[List[int]], t.Tokenizer]:
    tokenizer = t.Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer.texts_to_sequences(text), tokenizer


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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['loss'])
    model.fit(x=data, y=data, epochs=epochs, batch_size=batch_size)
    return model


if __name__ == '__main__':
    texts, text_count, text_len = load_data()
    unique_word_count = len(np.unique(texts))
    encoded_text, tokenizer = get_tokenizer(texts)
    tensor = tf.convert_to_tensor(encoded_text)

    input_length = len(encoded_text[0])

    autoencoder = get_seq_2_seq_autoencoder()
    autoencoder = train_autoencoder(autoencoder, batch_size=32, epochs=100)
