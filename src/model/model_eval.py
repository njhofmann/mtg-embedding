from typing import Tuple
import src.model.models.autoencoder as a
import src.model.data as d
from src.model.embeddings import RaggedIntArr


def eval_model(model: a.Seq2SeqAutoencoder, data: RaggedIntArr, max_len: int, folds: int) -> Tuple[float, float]:
    avg_loss, avg_accuracy = 0.0, 0.0
    for x_data, y_data, test_data in d.create_cross_valid_folds(data, max_len, folds):
        model.train(x_data, y_data)
        loss, accuracy = model.eval(test_data)
        avg_loss += loss
        avg_accuracy += avg_accuracy
    return avg_loss / folds, avg_accuracy / folds

