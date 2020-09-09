from typing import Tuple

import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow.keras.models as m


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


def create_seq_2_seq_autoencoder(embedding_len: int, word_count: int, sent_len: int, lstm_units: int) -> m.Model:
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


def train_autoencoder(model: m.Model, x: tf.Tensor, y: tf.Tensor, epochs: int, batch_size: int) -> m.Model:
    """Trains the given autoencoder model using the given training data, for the given number epochs at the listed
    batch size
    :param model: autoencoder to train
    :param x: input data
    :param y: true output data
    :param epochs: time to train model
    :param batch_size: samples to use at each training step
    :return: trained autoencoder
    """
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size)
    return model
