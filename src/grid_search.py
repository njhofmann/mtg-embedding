from typing import Dict, Iterator, Union, List, Optional, Tuple

HyperparamVals = Union[str, float, int]


def foo(unset_vals: List[Tuple[str, List[HyperparamVals]]], set_vals: Optional[Dict[str, HyperparamVals]] = None) \
        -> Iterator[Dict[str, HyperparamVals]]:
    if not set_vals:
        set_vals = {}

    if not unset_vals:
        yield set_vals
    else:
        cur_val = unset_vals[0]
        for val in cur_val[1]:
            temp_set_vals = {cur_val[0]: val, **set_vals}
            for item in foo(unset_vals[1:], temp_set_vals):
                yield item


def hyperparameter_grid_search(params: Dict[str, List[HyperparamVals]]) -> Iterator[Dict[str, HyperparamVals]]:
    """Given a dictionary of hyperparameters to possible values, performs grid search over those values by producing a
    generator that iterates over every possible combination of hyperparameter values as a dictionary of the
    hyperparameters to a specific value
    :param params: hyperparameter dictionary of all values
    :return: hyperparameter dictionary of specific values
    """
    for i in foo([(k, v) for k, v in params.items()]):
        yield i


if __name__ == '__main__':
    print(len(list(foo([('a', [1, 2, 3]), ('b', [4, 5, 6]), ('c', [7, 8, 9])]))))
