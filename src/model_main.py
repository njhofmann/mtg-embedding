import argparse as ap
import sys
from typing import Union, List

import src.util as u
from src.model import embeddings as e, models as m, data as d, model_eval as v
import src.paths as p

"""Program for training and evaluating different autoencoders based on user args"""


class ModelOptions(u.GenericOptions):
    Plain = 'plain'
    LSTM = 'lstm'
    LSTMAttention = 'lstm-attention'
    Transformer = 'transformer'


class DataOptions(u.GenericOptions):
    CardTexts = 'card-texts'
    CardTypes = 'card-type'
    ManaCosts = 'mana-costs'
    FlavorTexts = 'flavor-texts'


class TrainingRegimes(u.GenericOptions):
    Eval = 'eval'
    Final = 'final'
    EvalAndFinal = 'eval-and-final'


def get_model(model_type: ModelOptions, layers: Union[int, List[int]], embedding_len: int, vocab_size: int,
              sent_len: int) -> m.Seq2SeqAutoencoder:
    if model_type == ModelOptions.Plain and isinstance(layers, list):
        return m.PlainAutoencoder(input_len=sent_len, dense_sizes=layers)
    elif model_type == ModelOptions.LSTM and isinstance(layers, int):
        return m.LSTMAutoencoder(lstm_units=layers, input_embedding_len=embedding_len, vocab_size=vocab_size,
                                 sent_len=sent_len)
    elif model_type == ModelOptions.LSTMAttention:
        return m.LSTMAutoencoder(lstm_units=layers, input_embedding_len=embedding_len, vocab_size=vocab_size,
                                 sent_len=sent_len, attention=True)
    elif model_type == ModelOptions.Transformer:  # TODO transformer
        raise NotImplemented()
    raise ValueError('invalid model arguments')


def get_data(data_type: DataOptions) -> list:
    if data_type == DataOptions.CardTexts:
        path = p.CARD_TEXTS_PATH
    elif data_type == DataOptions.CardTypes:
        path = p.CARD_TYPES_PATH
    elif data_type == DataOptions.FlavorTexts:
        path = p.FLAVOR_TEXTS_PATH
    else:
        path = p.MANA_COSTS_PATH
    return p.load_pickle(path)


def parse_layer_option(arg: str) -> Union[List[int], int]:
    args = [int(i) for i in arg.split(' ')]
    return args[0] if len(args) == 1 else args


def get_embedding_len(embed_len: int, vocab_size: int) -> int:
    return embed_len if embed_len > 0 else e.optimal_embedding_len(vocab_size)


def get_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=ModelOptions.to_option, help='type of model to train')
    parser.add_argument('-d', '--data', required=True, type=DataOptions.to_option,
                        help='dataset to design an autoencoder around')
    parser.add_argument('-r', '--regime', required=True, type=TrainingRegimes.to_option,
                        help='training regime for model - evaluation and/or final model')
    parser.add_argument('-e', '--epochs', default=8, help='max epochs for training')
    parser.add_argument('-k', '--cross-val-folds', default=5, help='folds for cross validation')
    parser.add_argument('-l', '--layers', default=[32, 16], type=parse_layer_option,
                        help='num of nodes per layer for plain autoencoder, or LSTM units for LSTM architecture')
    parser.add_argument('-el', '--embedding_len', default=-1, type=int, help='embedding')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size for training models')
    return parser


if __name__ == '__main__':
    # TODO add embedding len as separate arg for plain autoencoder
    args = get_parser().parse_args(sys.argv[1:])

    if args.data == DataOptions.ManaCosts and args.model != ModelOptions.Plain:
        raise ValueError('can only train mana costs with a normal autoencoder')

    data = get_data(args.data)

    max_sent_len = e.get_longest_sent_size(data)
    unique_word_count = e.vocab_size(data) + 1
    encoded_texts, tokenizer = e.get_tokenizer(data)

    batch_size = args.batch_size
    epochs = args.epochs

    embed_len = get_embedding_len(args.embedding_len, unique_word_count)
    model = get_model(args.model, args.layers, embed_len, unique_word_count, max_sent_len)
    print(model.summary())

    evaluate = True
    final = True
    regime = args.regime
    if regime == TrainingRegimes.Eval:
        final = False
    elif regime == TrainingRegimes.Final:
        evaluate = False

    if evaluate:
        print(v.eval_model(model, encoded_texts, max_len=max_sent_len, folds=args.cross_val_folds, epochs=epochs,
                           batch_size=batch_size))

    if final:
        x_data, y_data = d.augment_data(encoded_texts)
        x_data = tokenizer.texts_to_sequences(x_data)
        y_data = tokenizer.texts_to_sequences(y_data)
        model.reset()
        model.train(x_data, y_data, batch_size=batch_size, epochs=epochs)
        model.save_model()
