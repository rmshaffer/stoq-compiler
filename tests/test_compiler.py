'''
Tests for stoqcompiler.compiler modules.
'''
import pytest
import numpy as np

from stoqcompiler.unitary import (
    Unitary,
    UnitaryPrimitive,
    UnitaryDefinitions,
    UnitarySequence,
    UnitarySequenceEntry,
    ParameterizedUnitary,
    ParameterizedUnitaryParameter)
from stoqcompiler.compiler import Compiler, CompilerAction

qubit_dimension = 2


class TestCompiler:
    def test_compile_no_unitary(self) -> None:
        compiler = Compiler(qubit_dimension, [])
        target_unitary = None
        with pytest.raises(Exception):
            compiler.compile(target_unitary)

    def test_compiler_action_enum(self) -> None:
        assert CompilerAction.is_append(CompilerAction.AppendFirst)
        assert CompilerAction.is_append(CompilerAction.AppendLast)
        assert not CompilerAction.is_append(CompilerAction.RemoveFirst)
        assert not CompilerAction.is_append(CompilerAction.RemoveLast)

        assert CompilerAction.is_remove(CompilerAction.RemoveFirst)
        assert CompilerAction.is_remove(CompilerAction.RemoveLast)
        assert not CompilerAction.is_remove(CompilerAction.AppendFirst)
        assert not CompilerAction.is_remove(CompilerAction.AppendLast)

    def test_compile_identity(self) -> None:
        unitary_primitives = [
            UnitaryPrimitive(Unitary.identity(qubit_dimension))]
        compiler = Compiler(qubit_dimension, unitary_primitives)

        num_qubits = 1
        system_dimension = qubit_dimension ** num_qubits
        target_unitary = Unitary.identity(system_dimension)
        result = compiler.compile(target_unitary)

        assert result.compiled_sequence.product().close_to(target_unitary)
        assert result.compiled_sequence.get_qasm()
        assert result.compiled_sequence.get_display_output()
        assert isinstance(result.cost_by_step, list)
        assert result.total_elapsed_time >= 0.0

    def test_compile_no_unitary_primitives(self) -> None:
        target_unitary = Unitary(qubit_dimension)
        with pytest.raises(Exception):
            compiler = Compiler(qubit_dimension)
            compiler.compile(target_unitary)

    def test_compile_sigmaz(self) -> None:
        system_dimension = qubit_dimension
        unitary_primitives = [
            UnitaryPrimitive(UnitaryDefinitions.rx(np.pi / 2)),
            UnitaryPrimitive(UnitaryDefinitions.ry(np.pi / 2))]
        compiler = Compiler(system_dimension, unitary_primitives)

        target_unitary = UnitaryDefinitions.sigmaz()
        result = compiler.compile(target_unitary)

        assert result.compiled_sequence.product().close_to(target_unitary)
        assert result.compiled_sequence.get_qasm()
        assert result.compiled_sequence.get_display_output()
        assert isinstance(result.cost_by_step, list)
        assert result.total_elapsed_time >= 0.0

    def test_compile_two_qubits(self) -> None:
        num_qubits = 2
        system_dimension = qubit_dimension ** num_qubits
        unitary_primitives = [
            UnitaryPrimitive(UnitaryDefinitions.rx(np.pi / 2)),
            UnitaryPrimitive(UnitaryDefinitions.cnot())]
        compiler = Compiler(system_dimension, unitary_primitives)

        # Ensure determinism by setting the random seed
        np.random.seed(12345)

        target_unitary = UnitarySequence(system_dimension, [
            UnitarySequenceEntry(UnitaryDefinitions.cnot(), [0, 1]),
            UnitarySequenceEntry(UnitaryDefinitions.rx(np.pi), [0])]).product()
        result = compiler.compile(target_unitary)

        assert result.compiled_sequence.product().close_to(target_unitary)
        assert result.compiled_sequence.get_qasm()
        assert result.compiled_sequence.get_display_output()
        assert isinstance(result.cost_by_step, list)
        assert result.total_elapsed_time >= 0.0

    def test_compile_sigmaz_approximate(self) -> None:
        threshold = 0.95

        def rotation_matrix(
            alpha: float,
            beta: float,
            gamma: float
        ) -> np.ndarray:
            return np.array(
                [[np.cos(beta / 2) * np.exp(-1j * (alpha + gamma) / 2),
                    -np.sin(beta / 2) * np.exp(-1j * (alpha - gamma) / 2)],
                 [np.sin(beta / 2) * np.exp(1j * (alpha - gamma) / 2),
                    np.cos(beta / 2) * np.exp(1j * (alpha + gamma) / 2)]])

        min_value = 0
        max_value = 2 * np.pi
        parameters = [ParameterizedUnitaryParameter(
                      "alpha", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter(
                      "beta", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter(
                      "gamma", min_value, max_value, is_angle=True)]
        display_name = "R"

        rotation = ParameterizedUnitary(
            qubit_dimension, rotation_matrix, parameters, display_name)

        unitary_primitives = [UnitaryPrimitive(rotation)]

        system_dimension = qubit_dimension
        compiler = Compiler(system_dimension, unitary_primitives)
        target_unitary = UnitaryDefinitions.sigmaz()
        result = compiler.compile(target_unitary, threshold)

        assert result.compiled_sequence.product().close_to(
            target_unitary, threshold)
        assert isinstance(result.cost_by_step, list)
        assert result.total_elapsed_time >= 0.0

        sequence_entries = result.compiled_sequence.get_sequence_entries()
        first_sequence_unitary = sequence_entries[0].get_full_unitary(
            system_dimension)
        assert first_sequence_unitary.get_parameter_value("bogus") is None
        for parameter in parameters:
            parameter_name = parameter.get_parameter_name()
            parameter_value = first_sequence_unitary.get_parameter_value(
                parameter_name)
            assert parameter_value is not None, parameter_name
