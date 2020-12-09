from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers as ly, models as m, optimizers as o

import src.model.models.autoencoder as a


class LSTMAutoencoder(a.Seq2SeqAutoencoder):

    def __init__(self, lstm_units: int, input_embedding_len: int, sent_len: int, vocab_size: int, learning_rate: float,
                 attention: bool = False) -> None:
        super(LSTMAutoencoder, self).__init__(learning_rate)
        # TODO add attention modules here
        self.lstm_units = lstm_units
        self.input_embedding_len = input_embedding_len
        self.sent_len = sent_len
        self.vocab_size = vocab_size
        self.reset()

    def compile(self) -> None:
        adam = o.Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def init_encoder(self) -> Tuple[m.Model, tf.Tensor, ly.Input]:
        encoder_inputs = ly.Input(shape=(self.sent_len,), name='encoder-input')
        embedded_inputs = ly.Embedding(input_dim=self.vocab_size, output_dim=self.input_embedding_len,
                                       input_length=self.sent_len, name='encoder-embedding',
                                       mask_zero=False)(encoder_inputs)
        # samples x timesteps x features
        state_h = ly.Bidirectional(ly.LSTM(self.lstm_units, activation='tanh', name='encoder-lstm'))(embedded_inputs)
        encoder = m.Model(inputs=encoder_inputs, outputs=state_h, name='encoder-model')
        encoder_outputs = encoder(encoder_inputs)
        return encoder, encoder_outputs, encoder_inputs

    def init_decoder(self, encoder_output: tf.Tensor) -> Tuple[m.Model, ly.Layer]:
        decoded = ly.RepeatVector(n=self.sent_len)(encoder_output)  # TODO what does this do
        decoder_lstm = ly.Bidirectional(ly.LSTM(self.lstm_units, return_sequences=True, name='decoder-dense'))(decoded)
        final_layer = ly.Dense(self.vocab_size, activation='softmax', name='decoder-dense')(decoder_lstm)
        decoder = None  # m.Model(ly.Input(shape=tuple(encoder_output.shape)), final_layer, name='decoder-model')
        # TODO fix me
        return decoder, final_layer

    def train(self, x: tf.Tensor, y: tf.Tensor, epochs: int, batch_size: int) -> None:
        self.autoencoder.fit(x=x, y=y, epochs=epochs, batch_size=batch_size)
