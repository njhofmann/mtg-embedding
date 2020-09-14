import random as r
from typing import List, Tuple, Generator, Iterator
from src.model.embeddings import RaggedIntArr
import tensorflow as tf
import tensorflow.keras.preprocessing.sequence as s
import numpy as np

"""Methods dealing with data related actions such as augmentation, train-test splitting, etc."""

PADDING_SYMBOL = 0


def remove_rand_item(items: List[int], del_prob: float) -> List[int]:
    return [PADDING_SYMBOL if r.random() < del_prob else item for item in items]


def swap_rand_items(items: List[int]) -> List[int]:
    idxs = r.sample(population=range(len(items)), k=2)
    new_items = [0] * len(items)
    for idx, item in enumerate(items):
        if idx == idxs[0]:
            item = items[idxs[1]]
        elif idx == idxs[1]:
            item = items[idxs[0]]
        new_items[idx] = item
    return new_items


def augment_data(data: RaggedIntArr, del_prob: float = .05, swap_count: int = 3) -> Tuple[RaggedIntArr, RaggedIntArr]:
    x_data, y_data = [], []
    for item in data:
        x_data.append(item)
        y_data.append(item)

        if (masked_item := remove_rand_item(item, del_prob)) != item:
            x_data.append(masked_item)
            y_data.append(item)

        if len(item) > 2:
            for i in range(swap_count):
                swapped_item = swap_rand_items(item)
                x_data.append(swapped_item)
                y_data.append(item)

    return x_data, y_data


def split_training_data(tokens: List[List[int]], split: float = .8) -> Tuple[RaggedIntArr, RaggedIntArr, RaggedIntArr]:
    r.shuffle(x=tokens)
    split_idx = round(len(tokens) * split)
    train, test = tokens[:split_idx], tokens[split_idx:]
    train_x, train_y = augment_data(train)
    return train_x, train_y, test


def create_cross_valid_folds(tokens: RaggedIntArr, max_len: int, folds: int) -> Iterator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    test_size = round(len(tokens) / folds)
    for i in range(folds):
        split_idx = i * test_size
        next_idx = split_idx + test_size
        test = tokens[split_idx:next_idx]
        train = tokens[0:split_idx] + tokens[next_idx:]
        train_x, train_y = augment_data(train)
        train_x = convert_to_tensor(train_x, max_len)
        train_y = convert_to_tensor(train_y, max_len)
        test = convert_to_tensor(test, max_len)
        yield train_x, train_y, test


def convert_to_tensor(ints: RaggedIntArr, max_len: int) -> tf.Tensor:
    padded = np.array(s.pad_sequences(ints, padding='post', maxlen=max_len))
    reshaped = np.reshape(np.array(padded), newshape=(*padded.shape, 1))
    return tf.convert_to_tensor(reshaped)
