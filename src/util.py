from model import embeddings as e
from src import options as o
from typing import Tuple


def eval_regime(regime: o.TrainingRegimes) -> Tuple[bool, bool]:
    evaluate, final = True, True
    if regime == o.TrainingRegimes.Eval:
        final = False
    elif regime == o.TrainingRegimes.Final:
        evaluate = False
    return evaluate, final


def get_embedding_len(embed_len: int, vocab_size: int) -> int:
    return embed_len if embed_len > 0 else e.optimal_embedding_len(vocab_size)