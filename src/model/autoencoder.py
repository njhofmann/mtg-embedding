import abc
from typing import Tuple, List
from src.model.embeddings import RaggedIntArr
import tensorflow as tf


class Seq2SeqAutoencoder(abc.ABC):

    @property
    @abc.abstractmethod
    def encoder(self):
        pass

    @abc.abstractmethod
    def init_encoder(self):
        pass

    @property
    @abc.abstractmethod
    def decoder(self):
        pass

    @property
    @abc.abstractmethod
    def init_decoder(self):
        pass

    @property
    @abc.abstractmethod
    def autoencoder(self):
        pass

    @property
    @abc.abstractmethod
    def init_autoencoder(self):
        pass

    @abc.abstractmethod
    def encode(self, words: RaggedIntArr) -> List[List[float]]:
        pass

    @abc.abstractmethod
    def train(self, x: tf.Tensor, y: tf.Tensor) -> None:
        pass

    @abc.abstractmethod
    def eval(self, x: tf.Tensor) -> Tuple[float, float]:
        pass