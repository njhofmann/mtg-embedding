from typing import Tuple, Iterable

import src.model.data as d
import src.model.models.autoencoder as a
from model.custom_types import RaggedIntArr


def eval_model(model: a.Seq2SeqAutoencoder, data: RaggedIntArr, max_len: int, folds: int, epochs: int, batch_size: int)\
        -> Tuple[float, float]:
    avg_loss, avg_accuracy = 0.0, 0.0
    for i, (x_data, y_data, test_data) in enumerate(d.gen_cross_valid_folds(data, folds)):
        x_data, y_data, test_data = d.convert_to_tensors(max_len, x_data, y_data, test_data)
        model.train(x_data, y_data, epochs, batch_size)
        loss, accuracy = model.eval(test_data)
        avg_loss += loss
        avg_accuracy += accuracy
        print(f'iteration {i}: loss {loss}, accuracy {accuracy}')
        model.reset()
    return avg_loss / folds, avg_accuracy / folds

