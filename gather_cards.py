import json as j
from typing import List, Dict
import re

REMINDER_TEXT_REGEX = re.compile('\\(.+\\)')
COST_REGEX = re.compile('{[0-9RGBWUC]+}')
SPLIT_REGEX = re.compile('[, .\n-]')
MANA_SYMBOLS_TO_IDX = {symbol: idx for idx, symbol in enumerate('WUBRGC')}


def remove_card_name(card_name: str, text: str) -> str:
    return text.replace(card_name, '~')


def format_string(string: str) -> List[str]:
    # remove reminder text
    string = REMINDER_TEXT_REGEX.sub('', string)  # TODO finish me
    print(string)
    # split
    words = SPLIT_REGEX.split(string)
    # split colons
    return [i.lower() for i in words if i]


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
    text = remove_card_name(name, card.get('oracle_text', ''))
    flavor_text = remove_card_name(name, card.get('flavor_text', ''))
    print(format_string(text))
    return {'name': name, 'mana_cost': mana_cost, 'text': text, 'flavor_text': flavor_text}


def parse_all_cards(json_path: str) -> None:
    with open(json_path, 'r') as f:
        raw_cards = j.load(f)
    parsed_cards = [parse_card(card) for card in raw_cards]


if __name__ == '__main__':
    parse_all_cards('oracle-cards-20200803170701.json')
