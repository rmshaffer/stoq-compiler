'''
Defines the CompilerResult class.
'''
from typing import List

from stoqcompiler.unitary import UnitarySequence


class CompilerResult:
    '''
    Represents the result of a STOQ compilation.
    '''
    def __init__(
        self,
        compiled_sequence: UnitarySequence,
        cost_by_step: List[float],
        total_elapsed_time: float
    ):
        '''
        Creates a CompilerResult object.

        :param compiled_sequence: The compiled sequence of unitaries.
        :type compiled_sequence: UnitarySequence
        :param cost_by_step: The value of the cost function (that is, the
        distance from the target unitary each compilation step.
        :type cost_by_step: List[float]
        :param total_elapsed_time: The total time taken to perform
        the compilation.
        :type total_elapsed_time: float
        '''
        self.compiled_sequence = compiled_sequence
        self.cost_by_step = cost_by_step
        self.total_elapsed_time = total_elapsed_time
