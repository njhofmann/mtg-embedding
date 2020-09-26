import argparse as ap

"""Program for training and evaluating different autoencoders based on user args"""


def get_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('-m', '--model', help='type of model to train')
    parser.add_argument('-d', '--data', help='dataset to use on autoencoder')
    parser.add_argument('-r', '--regime', help='training regime for model - evaluation and/or final model')
    return parser
