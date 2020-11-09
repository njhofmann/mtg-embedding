import enum as e


class GenericOptions(e.Enum):

    @classmethod
    def to_option(cls, str_option: str):
        for option in cls:
            if str_option == option.value:
                return option
        raise ValueError(f'invalid option {str_option} for {cls}, valid options are {[i for i in cls]}')
