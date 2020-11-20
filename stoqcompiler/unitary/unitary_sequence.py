'''
Defines the UnitarySequence class.
'''
import copy
import numpy as np
from typing import List

from .unitary import Unitary
from .unitary_sequence_entry import UnitarySequenceEntry


class UnitarySequence:
    '''
    Represents a sequence of unitaries applied to specific qubits
    in a system.
    '''
    def __init__(
        self,
        dimension: int,
        sequence_entries: List[UnitarySequenceEntry] = []
    ):
        '''
        Creates a UnitarySequence object.

        :param dimension: The dimension of the state space. For an n-qubit
            system, dimension should be set to 2**n.
        :type dimension: int
        :param sequence_entries: The entries of the unitary sequence,
            defaults to [].
        :type sequence_entries: List[UnitarySequenceEntry], optional
        '''
        self.dimension = dimension

        assert (isinstance(sequence_entries, list)
                or isinstance(sequence_entries, np.ndarray))
        [self._assert_is_entry_valid(entry) for entry in sequence_entries]

        self.sequence_entries = copy.deepcopy(sequence_entries)
        self.sequence_product = None

        self.previous_entries = None
        self.previous_sequence_product = None

    @classmethod
    def combine(
        this_class: 'UnitarySequence',
        *sequences: 'UnitarySequence'
    ) -> 'UnitarySequence':
        '''
        Returns the concatenation of two or more unitary sequences.

        :param this_class: The first unitary sequence.
        :type this_class: UnitarySequence
        :param sequences: The remanining unitary sequences.
        :type sequences: UnitarySequence
        :return: The concatenation of the sequences.
        :rtype: UnitarySequence
        '''
        assert len(sequences) > 0
        assert np.all([
            isinstance(sequence, this_class)
            for sequence in sequences])

        dimension = sequences[0].get_dimension()
        assert np.all([
            sequence.get_dimension() == dimension
            for sequence in sequences])

        new_sequence_entries = []
        new_product = Unitary.identity(dimension)
        for sequence in sequences:
            new_sequence_entries.extend(sequence.get_sequence_entries())
            new_product = new_product.left_multiply(sequence.product())

        new_sequence = this_class(dimension, new_sequence_entries)
        new_sequence.sequence_product = new_product
        return new_sequence

    def _assert_is_entry_valid(
        self,
        sequence_entry: UnitarySequenceEntry
    ) -> None:
        '''
        Validates that the specified sequence entry can be
        used with this sequence.
        '''
        assert isinstance(sequence_entry, UnitarySequenceEntry), \
            type(sequence_entry)
        assert sequence_entry.get_dimension() <= self.dimension
        assert 2**(np.max(sequence_entry.get_apply_to()) + 1) <= self.dimension

    def get_dimension(self) -> int:
        '''
        Gets the dimension of the state space on which
        this unitary sequence acts.

        :return: The state space dimension.
        :rtype: int
        '''
        return self.dimension

    def get_length(self) -> int:
        '''
        Gets the number of entries in this unitary sequence.

        :return: The sequence length.
        :rtype: int
        '''
        return len(self.sequence_entries)

    def get_sequence_entries(self) -> List[UnitarySequenceEntry]:
        '''
        Gets the entries of this unitary sequence.

        :return: The sequence entries.
        :rtype: List[UnitarySequenceEntry]
        '''
        return copy.deepcopy(self.sequence_entries)

    def _save_undo_state(self) -> None:
        '''
        Internal method to save the current state as the undo state.
        '''
        self.previous_entries = copy.deepcopy(self.sequence_entries)
        if self.sequence_product:
            self.previous_sequence_product = Unitary(
                self.sequence_product.get_dimension(),
                self.sequence_product.get_matrix())

    def append_first(
        self,
        sequence_entry: UnitarySequenceEntry,
        save_undo: bool = True
    ) -> None:
        '''
        Appends the given sequence entry as the first item in this sequence.

        :param sequence_entry: The sequence entry to append.
        :type sequence_entry: UnitarySequenceEntry
        :param save_undo: Whether to save the state prior to append as
            the undo state, defaults to True.
        :type save_undo: bool, optional
        '''
        self._assert_is_entry_valid(sequence_entry)

        if save_undo:
            self._save_undo_state()
        self.sequence_entries.insert(0, sequence_entry)
        if self.sequence_product:
            u_appended = sequence_entry.get_full_unitary(self.dimension)
            self.sequence_product = self.sequence_product.right_multiply(
                u_appended)

    def append_last(
        self,
        sequence_entry: UnitarySequenceEntry,
        save_undo: bool = True
    ) -> None:
        '''
        Appends the given sequence entry as the last item in this sequence.

        :param sequence_entry: The sequence entry to append.
        :type sequence_entry: UnitarySequenceEntry
        :param save_undo: Whether to save the state prior to append as
            the undo state, defaults to True.
        :type save_undo: bool, optional
        '''
        self._assert_is_entry_valid(sequence_entry)

        if save_undo:
            self._save_undo_state()
        self.sequence_entries.append(sequence_entry)
        if self.sequence_product:
            u_appended = sequence_entry.get_full_unitary(self.dimension)
            self.sequence_product = self.sequence_product.left_multiply(
                u_appended)

    def remove_first(
        self,
        save_undo: bool = True
    ) -> None:
        '''
        Removes the first sequence entry from this sequence.

        :param save_undo: Whether to save the state prior to remove as
            the undo state, defaults to True.
        :type save_undo: bool, optional
        '''
        if self.get_length() > 0:
            if save_undo:
                self._save_undo_state()
            entry_removed = self.sequence_entries.pop(0)
            if self.sequence_product:
                u_removed = entry_removed.get_full_unitary(self.dimension)
                self.sequence_product = self.sequence_product.right_multiply(
                    u_removed.inverse())

    def remove_last(
        self,
        save_undo: bool = True
    ) -> None:
        '''
        Removes the last sequence entry from this sequence.

        :param save_undo: Whether to save the state prior to remove as
            the undo state, defaults to True.
        :type save_undo: bool, optional
        '''
        if self.get_length() > 0:
            if save_undo:
                self._save_undo_state()
            entry_removed = self.sequence_entries.pop(-1)
            if self.sequence_product:
                u_removed = entry_removed.get_full_unitary(self.dimension)
                self.sequence_product = self.sequence_product.left_multiply(
                    u_removed.inverse())

    def undo(self) -> None:
        '''
        Revert to the most recent state when a change was made
        to this sequence with save_undo specified as True.
        '''
        assert self.previous_entries is not None, "can't undo"

        self.sequence_entries = self.previous_entries
        self.sequence_product = self.previous_sequence_product

        self.previous_entries = None
        self.previous_sequence_product = None

    def product(self) -> Unitary:
        '''
        Calculate the product of all entries in this sequence,
        where the first sequence entry is treated as the rightmost
        element of the product.

        :return: The unitary representing the sequence product.
        :rtype: Unitary
        '''
        if self.sequence_product is None:
            sequence_product = Unitary(self.dimension)
            for entry in self.sequence_entries:
                assert isinstance(entry, UnitarySequenceEntry)
                u_entry = entry.get_full_unitary(self.dimension)
                sequence_product = sequence_product.left_multiply(u_entry)
            self.sequence_product = sequence_product

        return self.sequence_product

    def inverse(self) -> 'UnitarySequence':
        '''
        Returns the inverse of this sequence, that is, the full sequence
        with the order of entries reversed and each unitary inverted.

        :return: The inverse sequence.
        :rtype: UnitarySequence.
        '''
        inverse_sequence = UnitarySequence(
            self.get_dimension(), self.get_sequence_entries())
        inverse_sequence.sequence_entries.reverse()
        for entry in inverse_sequence.sequence_entries:
            entry.unitary = entry.unitary.inverse()
            entry.full_unitary_by_dimension = {}
        inverse_sequence.sequence_product = self.product().inverse()
        return inverse_sequence

    def get_jaqal(self) -> str:
        '''
        Returns a JAQAL-like representation of this unitary sequence.
        This is a minimal-effort function and is not guaranteed to
        be a valid JAQAL program (and very likely will not be).

        :return: The JAQAL representation of this unitary sequence.
        :rtype: str
        '''
        return "// JAQAL generated from UnitarySequence.get_jaqal()\n" + \
            "\n".join([
                entry.get_full_unitary(self.get_dimension()).get_jaqal()
                for entry in self.sequence_entries])

    def get_qasm(self) -> str:
        '''
        Returns a QASM-like representation of this unitary sequence.
        This is a minimal-effort function and is not guaranteed to
        be a valid QASM program (and very likely will not be).

        :return: The QASM representation of this unitary sequence.
        :rtype: str
        '''
        return "# QASM generated from UnitarySequence.get_qasm()\n" + \
            "\n".join([
                entry.get_full_unitary(self.get_dimension()).get_qasm()
                for entry in self.sequence_entries])

    def get_display_output(self) -> str:
        '''
        Returns a human-readable representation of this unitary sequence.

        :return: A display representation of this unitary sequence.
        :rtype: str
        '''
        return "# Generated from UnitarySequence.get_display_output()\n" + \
            "\n".join([
                entry.get_full_unitary(self.get_dimension()).get_display_name()
                for entry in self.sequence_entries])
