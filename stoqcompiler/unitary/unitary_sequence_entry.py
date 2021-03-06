'''
Defines the UnitarySequenceEntry class.
'''
import numpy as np
import scipy.sparse
from typing import List

from .unitary import Unitary


class UnitarySequenceEntry:
    '''
    Represents an entry in a unitary sequence applied to
    a specific subset of qubits in a system.

    :param unitary: The unitary operation to be applied.
    :type unitary: Unitary
    :param apply_to: The qubits to which the operation is applied.
    :type apply_to: List[int]
    '''
    def __init__(
        self,
        unitary: Unitary,
        apply_to: List[int]
    ):
        '''
        Creates a UnitarySequenceEntry object.
        '''
        apply_to = list(apply_to)
        assert len(apply_to) == len(set(apply_to))
        assert 2**len(apply_to) == unitary.get_dimension()
        assert np.min(apply_to) >= 0

        self.unitary = unitary
        self.apply_to = apply_to
        self.full_unitary_by_dimension = {}

    def get_dimension(self) -> int:
        '''
        Gets the dimension of the state space on which
        this unitary acts.

        :return: The state space dimension.
        :rtype: int
        '''
        return self.unitary.get_dimension()

    def get_apply_to(self) -> List[int]:
        '''
        Gets the qubits to which the operation is applied.

        :return: The qubits to which the operation is applied.
        :rtype: List[int]
        '''
        return self.apply_to

    def get_permute_matrix(
        self,
        system_dimension: int
    ) -> np.ndarray:
        '''
        Function returning permute matrix for given system dimension.
        Adapted from qutip.permute._permute().

        :param system_dimension: The dimension of the state space
            of the system.
        :type system_dimension: int
        :return: The permute matrix.
        :rtype: np.ndarray
        '''
        assert system_dimension >= self.get_dimension()
        assert system_dimension >= 2**(np.max(self.get_apply_to()) + 1)

        def _select(
            sel: List[int],
            dims: List[int]
        ) -> np.ndarray:
            '''
            Private function finding selected components.
            Copied from qutip.ptrace._select().
            '''
            sel = np.asarray(sel)  # make sure sel is np.array
            dims = np.asarray(dims)  # make sure dims is np.array
            rlst = dims.take(sel)
            rprod = np.prod(rlst)
            ilist = np.ones((rprod, len(dims)), dtype=int)
            counter = np.arange(rprod)
            for k in range(len(sel)):
                ilist[:, sel[k]] = np.remainder(
                    np.fix(counter / np.prod(dims[sel[k + 1:]])),
                    dims[sel[k]]) + 1
            return ilist

        def _perm_inds(
            dims: List[int],
            order: List[int]
        ) -> np.ndarray:
            '''
            Private function giving permuted indices for permute function.
            Copied from qutip.permute._perm_inds().
            '''
            dims = np.asarray(dims)
            order = np.asarray(order)
            assert np.all(np.sort(order) == np.arange(len(dims))), \
                'Requested permutation does not match tensor structure.'
            sel = _select(order, dims)
            irev = np.fliplr(sel) - 1
            fact = np.append(np.array([1]), np.cumprod(np.flipud(dims)[:-1]))
            fact = fact.reshape(len(fact), 1)
            perm_inds = np.dot(irev, fact)
            return perm_inds

        num_system_qubits = int(np.log2(system_dimension))
        permute_order = [None] * num_system_qubits
        next_index = 0
        for apply in self.get_apply_to():
            permute_order[apply] = next_index
            next_index += 1
        for i in range(len(permute_order)):
            if permute_order[i] is None:
                permute_order[i] = next_index
                next_index += 1

        permute_indices = _perm_inds([2] * num_system_qubits, permute_order)
        data = np.ones(system_dimension, dtype=int)
        rows = np.arange(system_dimension, dtype=int)
        permute_matrix = scipy.sparse.coo_matrix(
            (data, (rows, np.array(permute_indices.T)[0])),
            shape=(system_dimension, system_dimension), dtype=int).tocsr()

        return permute_matrix

    def get_full_unitary(
        self,
        system_dimension: int
    ) -> Unitary:
        '''
        Gets the unitary operator acting on the space of the full system
        obtained by applying this unitary sequence entry to the specified
        subset of qubits.

        :param system_dimension: The dimension of the state space
            of the system.
        :type system_dimension: int
        :return: The unitary operator acting on the full system.
        :rtype: Unitary
        '''
        assert system_dimension >= self.get_dimension()

        if system_dimension not in self.full_unitary_by_dimension:
            expansion = Unitary.identity(
                int(system_dimension / self.get_dimension()))
            expanded_unitary = self.unitary.tensor(expansion)
            assert expanded_unitary.get_dimension() == system_dimension

            permute_matrix = self.get_permute_matrix(system_dimension)
            full_matrix = (permute_matrix
                           * expanded_unitary.get_matrix()
                           * permute_matrix.T)

            operation_name = self.unitary.get_operation_name()
            self.full_unitary_by_dimension[system_dimension] = Unitary(
                system_dimension,
                full_matrix,
                operation_name,
                self.unitary.parameter_dict,
                apply_to=self.get_apply_to())

        return self.full_unitary_by_dimension[system_dimension]
