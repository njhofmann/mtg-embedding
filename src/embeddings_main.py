import argparse as ap

"""Program for utilizing created autoencoders to create embeddings for whole MTG cards"""


def get_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('-ft', '--flavor-text', required=True, help='path to flavor text encoder')
    parser.add_argument('-cy', '--card-types', required=True, help='path to card type encoder')
    parser.add_argument('-mc', '--mana-costs', required=True, help='path to mana costs encoder')
    parser.add_argument('-ct', '--card-texts', required=True, help='path to card texts encoder')
    return parser
