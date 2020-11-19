from typing import List

from stoqcompiler.unitary import UnitarySequence


class CompilerResult:
    def __init__(
        self,
        compiled_sequence: UnitarySequence,
        cost_by_step: List[float],
        total_elapsed_time: float
    ):
        self.compiled_sequence = compiled_sequence
        self.cost_by_step = cost_by_step
        self.total_elapsed_time = total_elapsed_time
