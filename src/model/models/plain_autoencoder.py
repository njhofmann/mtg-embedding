from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras import models as m, layers as ly

import src.model.models.autoencoder as a


class PlainAutoencoder(a.Seq2SeqAutoencoder):

    def __init__(self, input_len: int, dense_sizes: List[int]):
        super(PlainAutoencoder, self).__init__()

        if not dense_sizes or PlainAutoencoder.in_descending_order(dense_sizes):
            raise ValueError('dense layer sizes must be in descending order')

        self.input_len = input_len
        self.dense_layers = dense_sizes

    @staticmethod
    def in_descending_order(nums: List[int]) -> bool:
        return len(nums) == 1 or all([nums[i] > nums[i + 1] for i in range(len(nums) - 1)])

    def create_recursive_dense_layers(self, input_layer: ly.Input, reverse: bool = False) -> ly.Dense:
        dense_layer = None
        layer_sizes = reversed(self.dense_layers) if reverse else self.dense_layers
        for layer_size in layer_sizes:
            dense_input = input_layer if dense_layer is None else dense_layer
            dense_layer = ly.Dense(units=layer_size, activation='relu')(dense_input)
        return dense_layer

    def init_encoder(self) -> Tuple[m.Model, ly.Input]:
        encoder_input = ly.Input(shape=(None, self.input_len))
        dense_layer = self.create_recursive_dense_layers(encoder_input)
        encoder = m.Model(encoder_input, dense_layer)
        return encoder, encoder_input

    def init_decoder(self, encoder_output: m.Model) -> Tuple[m.Model, ly.Layer]:
        dense_layer = self.create_recursive_dense_layers(encoder_output, reverse=True)
        decoder = m.Model(dense_layer)
        return decoder, dense_layer

    def train(self, x: tf.Tensor, y: tf.Tensor, epochs: int, batch_size: int) -> None:
        self.autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.autoencoder.fit(x=x, y=y, epochs=epochs, batch_size=batch_size)