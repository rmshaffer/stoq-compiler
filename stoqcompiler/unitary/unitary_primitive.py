'''
Defines the UnitaryPrimitive class.
'''
import numpy as np
from typing import List, Optional

from stoqcompiler.unitary import Unitary


class UnitaryPrimitive:
    '''
    Represents a primitive unitary operation and associated
    rules for applying it to a larger system of qubits.
    '''
    def __init__(
        self,
        unitary: Unitary,
        allowed_apply_to: Optional[List[List[int]]] = None
    ):
        '''
        Creates a UnitaryPrimitive object.

        :param unitary: The unitary operation.
        :type unitary: Unitary
        :param allowed_apply_to: All lists of qubits to which this
            operation is allowed to be applied, defaults to None.
            If not specified, all qubits are assumed to be allowed.
        :type allowed_apply_to: Optional[List[List[int]]], optional
        '''
        if allowed_apply_to is not None:
            allowed_apply_to = list(allowed_apply_to)
            assert np.all([
                len(apply_to) == len(set(apply_to))
                for apply_to in allowed_apply_to])
            assert np.all([
                2**len(apply_to) == unitary.get_dimension()
                for apply_to in allowed_apply_to])
            assert np.all([
                np.min(apply_to) >= 0
                for apply_to in allowed_apply_to])

        self.unitary = unitary
        self.allowed_apply_to = allowed_apply_to

    def get_unitary(self) -> Unitary:
        '''
        Gets the unitary operation.

        :return: The unitary operation.
        :rtype: Unitary
        '''
        return self.unitary

    def get_allowed_apply_to(self) -> Optional[List[List[int]]]:
        '''
        Gets all lists of qubits to which this operation is allowed
        to be applied. A return value of None means that all qubits
        are assumed to be allowed.

        :return: All lists of qubits to which this operation is allowed
            to be applied.
        :rtype: Optional[List[List[int]]]
        '''
        return self.allowed_apply_to
