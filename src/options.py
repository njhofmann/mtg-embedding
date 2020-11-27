import enum as e
from typing import Union, List

import paths as p
from model import models as m


class GenericOptions(e.Enum):

    @classmethod
    def to_option(cls, str_option: str):
        for option in cls:
            if str_option == option.value:
                return option
        raise ValueError(f'invalid option {str_option} for {cls}, valid options are {[i for i in cls]}')


class ModelOptions(GenericOptions):
    Plain = 'plain'
    LSTM = 'lstm'
    LSTMAttention = 'lstm-attention'
    Transformer = 'transformer'


class DataOptions(GenericOptions):
    CardTexts = 'card-texts'
    CardTypes = 'card-type'
    ManaCosts = 'mana-costs'
    FlavorTexts = 'flavor-texts'


class TrainingRegimes(GenericOptions):
    Eval = 'eval'
    Final = 'final'
    EvalAndFinal = 'eval-and-final'


def load_data(data_type: DataOptions) -> list:
    if data_type == DataOptions.CardTexts:
        path = p.CARD_TEXTS_PATH
    elif data_type == DataOptions.CardTypes:
        path = p.CARD_TYPES_PATH
    elif data_type == DataOptions.FlavorTexts:
        path = p.FLAVOR_TEXTS_PATH
    else:
        path = p.MANA_COSTS_PATH
    return p.load_pickle(path)


def init_model(model_type: ModelOptions, layers: Union[int, List[int]], embedding_len: int, vocab_size: int,
               sent_len: int) -> m.Seq2SeqAutoencoder:
    if model_type == ModelOptions.Plain and isinstance(layers, list):
        return m.PlainAutoencoder(input_len=sent_len, dense_sizes=layers)
    elif model_type == ModelOptions.LSTM and isinstance(layers, int):
        return m.LSTMAutoencoder(lstm_units=layers, input_embedding_len=embedding_len, vocab_size=vocab_size,
                                 sent_len=sent_len)
    elif model_type == ModelOptions.LSTMAttention:
        return m.LSTMAutoencoder(lstm_units=layers, input_embedding_len=embedding_len, vocab_size=vocab_size,
                                 sent_len=sent_len, attention=True)
    elif model_type == ModelOptions.Transformer:  # TODO transformer
        raise NotImplemented()
    raise ValueError('invalid model arguments')
