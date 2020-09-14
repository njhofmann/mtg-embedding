import pathlib as pl
import pickle as p

PICKLE_DIRC = pl.Path(__file__).parent.parent.joinpath('pickle-files')
CARD_NAMES_PATH = PICKLE_DIRC.joinpath('card_names.pickle')
MANA_COSTS_PATH = PICKLE_DIRC.joinpath('mana_costs.pickle')
CARD_TYPES_PATH = PICKLE_DIRC.joinpath('card_types.pickle')
CARD_TEXTS_PATH = PICKLE_DIRC.joinpath('card_texts.pickle')
FLAVOR_TEXTS_PATH = PICKLE_DIRC.joinpath('flavor_texts.pickle')


def load_pickle(path: pl.Path) -> list:
    with open(path, 'rb') as f:
        return p.load(f)
