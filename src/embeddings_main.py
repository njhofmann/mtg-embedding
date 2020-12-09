import argparse as ap
import tensorflow.keras.models as m
import sys
from typing import Optional
import src.paths as p
import numpy as np
import pathlib as pl

"""Program for utilizing created autoencoders to create embeddings for whole MTG cards"""


def get_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('-ft', '--flavor_text', required=True, help='path to flavor text encoder')
    parser.add_argument('-cy', '--card_types', required=True, help='path to card type encoder')
    parser.add_argument('-mc', '--mana_costs', required=True, help='path to mana costs encoder')
    parser.add_argument('-ct', '--card_texts', required=True, help='path to card texts encoder')
    return parser


def load_model_and_predict(model_path: str, data_path: pl.Path, tokenizer_path: Optional[pl.Path]) -> np.ndarray:
    data = p.load_pickle(data_path)
    if tokenizer_path:
        tokenizer = p.load_pickle(tokenizer_path)

    # TODO save & load tokenizers, tokenize data, convert to tensor
    return m.load_model(model_path).predict()


if __name__ == '__main__':
    args = get_parser().parse_args(sys.argv[1:])

    flavor_text_embeddings = load_model_and_predict(args.flavor_text, p.FLAVOR_TEXTS_PATH)
    card_type_embeddings = load_model_and_predict(args.card_types, p.CARD_TYPES_PATH)
    card_text_embeddings = load_model_and_predict(args.card_texts, p.CARD_TEXTS_PATH)
    mana_costs_embeddings = load_model_and_predict(args.mana_costs, p.MANA_COSTS_PATH)

    combined_embeddings = np.concatenate([mana_costs_embeddings, card_type_embeddings, card_text_embeddings,
                                          flavor_text_embeddings], axis=1)
    np.save(combined_embeddings, )