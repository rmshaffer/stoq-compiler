from stoqcompiler.verification import Verification
from stoqcompiler.compiler import CompilerResult
from stoqcompiler.unitary import (
    Unitary,
    UnitaryPrimitive,
    ParameterizedUnitaryDefinitions)

qubit_dimension = 2


class TestVerification:

    def test_rav(self) -> None:
        num_system_qubits = 2
        system_dimension = qubit_dimension ** num_system_qubits
        unitary_primitives = [
            UnitaryPrimitive(ParameterizedUnitaryDefinitions.rotation_xy()),
            UnitaryPrimitive(ParameterizedUnitaryDefinitions.xx())]

        sequence_length = 10
        threshold = 0.5
        rav_result = Verification.generate_rav_sequence(
            system_dimension, unitary_primitives, sequence_length, threshold)
        assert isinstance(rav_result, CompilerResult)

        product = rav_result.compiled_sequence.product()
        assert product.close_to(
            Unitary.identity(system_dimension), threshold), \
            product.distance_from(Unitary.identity(system_dimension))
        assert rav_result.compiled_sequence.get_qasm()
        assert rav_result.compiled_sequence.get_jaqal()

    def test_layered_rav(self) -> None:
        num_system_qubits = 2
        system_dimension = qubit_dimension ** num_system_qubits
        unitary_primitive_counts = {
            UnitaryPrimitive(ParameterizedUnitaryDefinitions.rotation_xy()): 3,
            UnitaryPrimitive(ParameterizedUnitaryDefinitions.xx()): 1
        }

        layer_count = 10
        threshold = 0.5
        layered_rav_result = Verification.generate_layered_rav_sequence(
            system_dimension, unitary_primitive_counts, layer_count, threshold)
        assert isinstance(layered_rav_result, CompilerResult)

        layer_length = sum(unitary_primitive_counts.values())
        assert len(
            layered_rav_result.compiled_sequence.get_sequence_entries()
        ) % layer_length == 0

        product = layered_rav_result.compiled_sequence.product()
        assert product.close_to(
            Unitary.identity(system_dimension), threshold), \
            product.distance_from(Unitary.identity(system_dimension))
        assert layered_rav_result.compiled_sequence.get_qasm()
        assert layered_rav_result.compiled_sequence.get_jaqal()
