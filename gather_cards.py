import json as j
from typing import List, Dict
import re

REMINDER_TEXT_REGEX = re.compile('\\(.+\\)')
COST_REGEX = re.compile('{[0-9RGBWUC]+}')
SPLIT_REGEX = re.compile('[, .\n]')
MANA_SYMBOLS_TO_IDX = {symbol: idx for idx, symbol in enumerate('WUBRGC')}
SAVE_SPLIT_REGEX = re.compile('(}{)|:')
TYPES_SPLIT_REGEX = re.compile('(\ —\ | )')


def remove_card_name(card_name: str, text: str) -> str:
    return text.replace(card_name, '~')


def format_string(string: str, card_name: str) -> List[str]:
    string = remove_card_name(card_name, string)
    string = REMINDER_TEXT_REGEX.sub('', string)

    for exp, replace in (':', ' :'), ('}{', '} {'):
        string = string.replace(exp, replace)

    words = SPLIT_REGEX.split(string)
    return lowercase_sentence(words)


def lowercase_sentence(stnc: List[str]) -> List[str]:
    return [i.lower() for i in stnc if i]


def format_types(type_line: str) -> List[str]:
    types = TYPES_SPLIT_REGEX.split(type_line)
    return lowercase_sentence(types)

def format_mana_cost(mana_cost: str) -> List[int]:
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


def parse_card(card) -> dict:
    name = card['name']
    mana_cost = format_mana_cost(card.get('mana_cost', ''))
    types = format_types(card.get('type_line', ''))
    text = format_string(card.get('oracle_text', ''), name)
    flavor_text = format_string(card.get('flavor_text', ''), name)
    return {'name': name, 'mana_cost': mana_cost, 'types': types, 'text': text, 'flavor_text': flavor_text}


def parse_all_cards(json_path: str) -> None:
    with open(json_path, 'r') as f:
        raw_cards = j.load(f)
    parsed_cards = [parse_card(card) for card in raw_cards]
    print(parsed_cards)


if __name__ == '__main__':
    parse_all_cards('oracle-cards-20200803170701.json')
