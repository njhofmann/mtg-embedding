import json as j
import pickle as p
import re
from typing import List

import numpy as np

import src.paths as c

"""Script for converts raw MTG data into standard format for future processing"""

UN_SETS = ['Unsanctioned', 'Unhinged', 'Unstable', 'Unglued']
REMINDER_TEXT_REGEX = re.compile('\\(.+\\)')
COST_REGEX = re.compile('{[0-9RGBWUC]+}')
SPLIT_REGEX = re.compile('[, .\n/]')
MANA_SYMBOLS_TO_IDX = {symbol: idx for idx, symbol in enumerate('WUBRGC')}
SAVE_SPLIT_REGEX = re.compile('(}{)|:')
TYPES_SPLIT_REGEX = re.compile('[( — ) ]')
FLAVOR_TEXT_REGEX = re.compile('["!—?]')
REMOVE_REGEX = re.compile('[•—]')


def pad_jagged_matrix(matrix: List[List[str]]) -> np.ndarray:
    """Pads the inner arrays of the given matrix of strings with NaN such that each array is of length equal
    to the largest array in the matrix. Then converts the array to a numpy array
    :param matrix: matrix of strings
    :return: square numpy array of strings
    """
    return np.append(arr=np.array([], dtype=np.str), values=[np.array(row, dtype=np.str) for row in matrix])
    # max_array_length = max(map(len, matrix))
    # matrix = [array + [np.NaN] * (max_array_length - len(array)) for array in matrix]
    # return np.array(matrix)


def remove_card_name(card_name: str, text: str) -> str:
    """If the given card name is present in the given card text, replaces it with a tilde
    :param card_name: name of card to look for
    :param text: card text
    :return: card text with any card name mentioned replaced with tilde
    """
    return text.replace(card_name, '~')


def format_string(string: str, card_name: str) -> List[str]:
    """Formats the given string by removing any referees to the given card name, removing illegal characters, splitting
    along any designated split characters / phrases, etc.
    :param string: string to format
    :param card_name: name of card
    :return: formatted string
    """
    string = remove_card_name(card_name, string)
    string = REMOVE_REGEX.sub('', string)
    string = REMINDER_TEXT_REGEX.sub('', string)

    for exp, replace in (':', ' :'), ('}{', '} {'):
        string = string.replace(exp, replace)

    words = SPLIT_REGEX.split(string)
    return lowercase_sentence(words)


def lowercase_sentence(stnc: List[str]) -> List[str]:
    """Lowercases each word in the given list of words, removing all empty words
    :param stnc: list of strings
    :return: list of words lowercased
    """
    return [i.lower() for i in stnc if i]


def format_card_types(card_types: str) -> List[str]:
    """Formats the given card types by splitting along valid card type characters
    :param card_types: card types
    :return: type_line clean card types
    """
    types = TYPES_SPLIT_REGEX.split(card_types)
    return lowercase_sentence(types)


def format_flavor_text(flavor: List[str]) -> List[str]:
    """Formats the given flavor text by removing all illegal flavor text characters
    :param flavor: flavor text
    :return: cleaned flavor text
    """
    flavor = [FLAVOR_TEXT_REGEX.sub('', i) for i in flavor]
    return [i for i in flavor if i]


def format_mana_cost(mana_cost: str) -> List[int]:
    """Formats the given mana cost string (ex: '{R}', '{2}{R}{G}', {W|{W}{U}', etc') into a standard representation.
    Represents each mana cost as a 7 element list of integers where each element corresponds to the number of times a
    specific mana type appeared in the mana cost.

     [W U B R G colorless wastes]

    :param mana_cost: raw card mana cost
    :return: given mana cost in standard representation
    """
    # format as 7 item list - W U B R G colorless wastes
    results = COST_REGEX.findall(mana_cost)
    counts = [0 for i in range(7)]
    for result in results:
        result = result[1:-1]
        try:
            counts[MANA_SYMBOLS_TO_IDX[result]] += 1
        except KeyError:
            counts[-1] += int(result)
    return counts


def parse_card(card_info: dict) -> dict:
    """Parses the given package of card info into a standard package of formatted card name, mana cost, card types,
    and flavor text
    :param card_info: raw card info
    :return: formatted card info
    """
    name = card_info['name']
    mana_cost = format_mana_cost(card_info.get('mana_cost', ''))
    types = format_card_types(card_info.get('type_line', ''))
    text = format_string(card_info.get('oracle_text', ''), name)
    flavor_text = format_flavor_text(format_string(card_info.get('flavor_text', ''), name))
    return {'name': name, 'mana_cost': mana_cost, 'types': types, 'text': text, 'flavor_text': flavor_text}


def parse_cards(card: dict) -> List[dict]:
    """Parses the given dictionary of card(s) info into a standard format, returns a dictionary for each card
    :param card: card info to parse, may have multiple cards
    :return: list of parsed card info
    """
    if ' // ' in card['name']:
        return [parse_card(card) for card in card['card_faces']]
    # don't format tokens or cards from unsets
    elif 'Token' in card['type_line'] or any([unset == card['set_name'] for unset in UN_SETS]):
        return []
    return [parse_card(card)]


def parse_all_cards(json_path: str) -> None:
    """For each instance of card info within the given JSON file, parses it into standard format and saves the resulting
    data
    :param json_path: path to JSON object of card info instances
    :return: None
    """
    with open(json_path, 'r') as f:
        raw_cards = j.load(f)
    parsed_cards = []
    for card in raw_cards:
        parsed_cards.extend(parse_cards(card))

    empty_list = lambda: [None for _ in range(len(parsed_cards))]
    names = empty_list()
    mana_costs = empty_list()
    types = empty_list()
    texts = empty_list()
    flavor_texts = empty_list()
    for idx, card in enumerate(parsed_cards):
        names[idx] = card['name']
        mana_costs[idx] = card['mana_cost']
        types[idx] = card['types']
        texts[idx] = card['text']
        flavor_texts[idx] = card['flavor_text']

    for lst, name in (names, c.CARD_NAMES_PATH), (mana_costs, c.MANA_COSTS_PATH), (types, c.CARD_TYPES_PATH), \
                     (texts, c.CARD_TEXTS_PATH), (flavor_texts, c.FLAVOR_TEXTS_PATH):
        with open(name, 'wb') as f:
            p.dump(lst, f)


if __name__ == '__main__':
    parse_all_cards('../oracle-cards-20200803170701.json')
