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


if __name__ == '__main__':
    texts, text_count, text_len = load_data()
    unique_word_count = len(np.unique(texts))
    encoded_text, tokenizer = get_tokenizer(texts)
    tensor = tf.convert_to_tensor(encoded_text)

    input_length = len(encoded_text[0])

    model = m.Sequential([
        l.Embedding(input_length=text_len, input_dim=unique_word_count + 1, output_dim=32),
        l.Dense(units=256, activation='relu'),
        l.Dense(units=1)
    ])
    print(model.summary())

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(x=tensor, y=tensor, epochs=32)

    print(tokenizer.texts_to_sequences(texts))
    print(len(np.unique(texts)))
    print(tokenizer.sequences_to_texts([[5, 6, 98]]))
