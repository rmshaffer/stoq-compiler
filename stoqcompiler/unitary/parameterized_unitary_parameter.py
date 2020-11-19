import numpy as np


class ParameterizedUnitaryParameter:

    def __init__(
        self,
        parameter_name: str,
        min_value: float,
        max_value: float,
        is_angle: bool
    ):
        assert parameter_name
        assert max_value >= min_value

        self.parameter_name = parameter_name
        self.min_value = min_value
        self.max_value = max_value
        self.is_angle = is_angle

    def get_parameter_name(self) -> str:
        return self.parameter_name

    def get_is_angle(self) -> bool:
        return self.is_angle

    def is_valid(
        self,
        parameter_value: float
    ) -> bool:
        return (parameter_value >= self.min_value
                and parameter_value <= self.max_value)

    def random_value(self) -> float:
        return (np.random.random_sample()
                * (self.max_value - self.min_value)) + self.min_value
