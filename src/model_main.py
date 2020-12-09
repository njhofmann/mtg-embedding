import argparse as ap
import sys
from typing import Union, List

from src import options as o, parser as p, util as u
from src.model import embeddings as e, data as d, model_eval as v
from util import get_embedding_len

"""Program for training and evaluating different autoencoders based on user args"""


def parse_layer_option(arg: str) -> Union[List[int], int]:
    args = [int(i) for i in arg.split(' ')]
    return args[0] if len(args) == 1 else args


def get_parser() -> ap.ArgumentParser:
    parser = p.init_common_parser()
    parser.add_argument('-e', '--epochs', default=8, type=int, help='max epochs for training')
    parser.add_argument('-l', '--layers', default=[32, 16], type=parse_layer_option, nargs='+',
                        help='num of nodes per layer for plain autoencoder, or LSTM units for LSTM architecture')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size for training models')
    parser.add_argument('-lr', '--learning_rate', type=float, help='ADAM learning rate')
    return parser


if __name__ == '__main__':
    # TODO add embedding len as separate arg for plain autoencoder
    args = get_parser().parse_args(sys.argv[1:])

    if args.data == o.DataOptions.ManaCosts and args.model != o.ModelOptions.Plain:
        raise ValueError('can only train mana costs with a normal autoencoder')

    data = o.load_data(args.data)

    max_sent_len = e.get_longest_sent_size(data)
    unique_word_count = e.vocab_size(data) + 1
    encoded_texts, tokenizer = e.get_tokenizer(data)

    batch_size = args.batch_size
    epochs = args.epochs

    embed_len = get_embedding_len(args.embedding_len, unique_word_count)
    layers = args.layers[0] if len(args.layers) == 1 else args.layers
    model = o.init_model(args.model, layers, embed_len, unique_word_count, max_sent_len, args.learning_rate)
    print(model.summary())

    evaluate, final = u.eval_regime(args.regime)

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
