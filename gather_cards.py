import json as j
from typing import List, Dict


def remove_card_name(card_name: str, text: str) -> str:
    return text.replace(card_name, '~')


def format_mana_cost(mana_cost: str) -> List[str]:
    return []


def parse_card(card) -> dict:
    name = card['name']
    mana_cost = card.get('mana_cost', '')
    text = card.get('oracle_text', '')
    flavor_text = card.get('flavor_text', '')
    return {'name': name, 'mana_cost': mana_cost, 'text': text, 'flavor_text': flavor_text}


def parse_all_cards(json_path: str) -> None:
    with open(json_path, 'r') as f:
        raw_cards = j.load(f)
    parsed_cards = [parse_card(card) for card in raw_cards]
    print(parsed_cards)


if __name__ == '__main__':
    parse_all_cards('oracle-cards-20200803170701.json')
