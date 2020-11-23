from src import options as o
from typing import Tuple


def eval_regime(regime: o.TrainingRegimes) -> Tuple[bool, bool]:
    evaluate, final = True, True
    if regime == o.TrainingRegimes.Eval:
        final = False
    elif regime == o.TrainingRegimes.Final:
        evaluate = False
    return evaluate, final
