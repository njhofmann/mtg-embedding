import abc
from typing import Tuple, List

import tensorflow as tf
import tensorflow.keras.backend as b
import tensorflow.keras.layers as ly
import tensorflow.keras.models as m
import tensorflow.keras.preprocessing.text as t

import src.model.data as d


class Seq2SeqAutoencoder(abc.ABC):

    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    def reset(self) -> None:
        b.clear_session()
        del self.encoder, self.decoder, self.autoencoder
        self.encoder, encoder_input = self.init_encoder()
        self.decoder, decoder_output = self.init_decoder(self.encoder)
        self.autoencoder = self.init_autoencoder(encoder_input, decoder_output)

    @abc.abstractmethod
    def init_encoder(self) -> Tuple[m.Model, ly.Input]:
        pass

    @abc.abstractmethod
    def init_decoder(self, encoder_output: m.Model) -> Tuple[m.Model, ly.Layer]:
        pass

    def init_autoencoder(self, encoder_input: ly.Input, decoder_output: ly.Layer) -> m.Model:
        return m.Model(encoder_input, decoder_output)

    def encode(self, words: List[List[str]], tokenizer: t.Tokenizer, max_len: int) -> List[List[float]]:
        data = d.convert_to_tensor(tokenizer.texts_to_sequences(words), max_len)
        return self.encoder.predict(x=data)

    @abc.abstractmethod
    def train(self, x: tf.Tensor, y: tf.Tensor, epochs: int, batch_size: int) -> None:
        pass

    def eval(self, test_data: tf.Tensor) -> Tuple[float, float]:
        return self.autoencoder.evaluate(x=test_data, y=test_data)
