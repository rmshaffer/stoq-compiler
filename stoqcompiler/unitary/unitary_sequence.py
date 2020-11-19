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
    Represents a sequence of unitary objects.
    '''
    def __init__(
        self,
        dimension: int,
        sequence_entries: List[UnitarySequenceEntry] = []
    ):
        self.dimension = dimension

        assert (isinstance(sequence_entries, list)
                or isinstance(sequence_entries, np.ndarray))
        [self.assert_is_entry_valid(entry) for entry in sequence_entries]

        self.sequence_entries = copy.deepcopy(sequence_entries)
        self.sequence_product = None

        self.previous_entries = None
        self.previous_sequence_product = None

    @classmethod
    def combine(
        this_class: 'UnitarySequence',
        *sequences: 'UnitarySequence'
    ) -> 'UnitarySequence':
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

    def assert_is_entry_valid(
        self,
        sequence_entry: UnitarySequenceEntry
    ) -> None:
        assert isinstance(sequence_entry, UnitarySequenceEntry), \
            type(sequence_entry)
        assert sequence_entry.get_dimension() <= self.dimension
        assert 2**(np.max(sequence_entry.get_apply_to()) + 1) <= self.dimension

    def get_dimension(self) -> int:
        return self.dimension

    def get_length(self) -> int:
        return len(self.sequence_entries)

    def get_sequence_entries(self) -> List[UnitarySequenceEntry]:
        return copy.deepcopy(self.sequence_entries)

    def save_undo_state(self) -> None:
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
        self.assert_is_entry_valid(sequence_entry)

        if save_undo:
            self.save_undo_state()
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
        self.assert_is_entry_valid(sequence_entry)

        if save_undo:
            self.save_undo_state()
        self.sequence_entries.append(sequence_entry)
        if self.sequence_product:
            u_appended = sequence_entry.get_full_unitary(self.dimension)
            self.sequence_product = self.sequence_product.left_multiply(
                u_appended)

    def remove_first(
        self,
        save_undo: bool = True
    ) -> None:
        if self.get_length() > 0:
            if save_undo:
                self.save_undo_state()
            entry_removed = self.sequence_entries.pop(0)
            if self.sequence_product:
                u_removed = entry_removed.get_full_unitary(self.dimension)
                self.sequence_product = self.sequence_product.right_multiply(
                    u_removed.inverse())

    def remove_last(
        self,
        save_undo: bool = True
    ) -> None:
        if self.get_length() > 0:
            if save_undo:
                self.save_undo_state()
            entry_removed = self.sequence_entries.pop(-1)
            if self.sequence_product:
                u_removed = entry_removed.get_full_unitary(self.dimension)
                self.sequence_product = self.sequence_product.left_multiply(
                    u_removed.inverse())

    def undo(self) -> None:
        assert self.previous_entries is not None, "can't undo"

        self.sequence_entries = self.previous_entries
        self.sequence_product = self.previous_sequence_product

        self.previous_entries = None
        self.previous_sequence_product = None

    def product(self) -> Unitary:
        if self.sequence_product is None:
            sequence_product = Unitary(self.dimension)
            for entry in self.sequence_entries:
                assert isinstance(entry, UnitarySequenceEntry)
                u_entry = entry.get_full_unitary(self.dimension)
                sequence_product = sequence_product.left_multiply(u_entry)
            self.sequence_product = sequence_product

        return self.sequence_product

    def inverse(self) -> 'UnitarySequence':
        inverse_sequence = UnitarySequence(
            self.get_dimension(), self.get_sequence_entries())
        inverse_sequence.sequence_entries.reverse()
        for entry in inverse_sequence.sequence_entries:
            entry.unitary = entry.unitary.inverse()
            entry.full_unitary_by_dimension = {}
        inverse_sequence.sequence_product = self.product().inverse()
        return inverse_sequence

    def get_jaqal(self) -> str:
        return "// JAQAL generated from UnitarySequence.get_jaqal()\n" + \
            "\n".join([
                entry.get_full_unitary(self.get_dimension()).get_jaqal()
                for entry in self.sequence_entries])

    def get_qasm(self) -> str:
        return "# QASM generated from UnitarySequence.get_qasm()\n" + \
            "\n".join([
                entry.get_full_unitary(self.get_dimension()).get_qasm()
                for entry in self.sequence_entries])

    def get_display_output(self) -> str:
        return "# Generated from UnitarySequence.get_display_output()\n" + \
            "\n".join([
                entry.get_full_unitary(self.get_dimension()).get_display_name()
                for entry in self.sequence_entries])
