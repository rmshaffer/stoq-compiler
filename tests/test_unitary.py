import pytest
import numpy as np

from stoqcompiler.unitary import (
    Unitary,
    UnitarySequence,
    UnitarySequenceEntry,
    UnitaryDefinitions,
    ParameterizedUnitary,
    ParameterizedUnitaryParameter,
    ParameterizedUnitaryDefinitions)


class TestUnitary:

    def test_default(self):
        dimension = 4
        unitary = Unitary(dimension)

        assert unitary.get_dimension() == dimension
        assert unitary.close_to(np.identity(dimension))

    def test_non_square_matrix(self):
        dimension = 2
        with pytest.raises(Exception):
            Unitary(dimension, np.array([[1, 0, 0], [0, 1, 0]]))

    def test_non_unitary_matrix(self):
        dimension = 2
        with pytest.raises(Exception):
            Unitary(dimension, np.array([[1, 0], [1, 1]]))

    def test_mismatched_dimension(self):
        dimension = 4
        with pytest.raises(Exception):
            Unitary(dimension, np.identity(dimension - 1))

    def test_inverse_fixed(self):
        dimension = 2
        operation_name = 'U'
        unitary = Unitary(dimension, np.array([
            [np.exp(1j*np.pi/4), 0],
            [0, 1j]]), operation_name)
        inverse = unitary.inverse()

        assert unitary.left_multiply(inverse).close_to(np.identity(dimension))
        assert unitary.right_multiply(inverse).close_to(np.identity(dimension))

        assert inverse.get_display_name() == 'U†'
        double_inverse = inverse.inverse()
        assert double_inverse.get_display_name() == 'U'
        assert unitary.close_to(double_inverse)

    def test_inverse_random(self):
        dimension = 2
        unitary = Unitary.random(dimension)
        inverse = unitary.inverse()

        assert unitary.left_multiply(inverse).close_to(np.identity(dimension))
        assert unitary.right_multiply(inverse).close_to(np.identity(dimension))

    def test_tensor(self):
        dimension = 2
        unitary = Unitary(dimension)
        tensor_product = unitary.tensor(unitary)

        assert tensor_product.get_dimension() == dimension ** 2
        assert tensor_product.close_to(np.identity(dimension ** 2))

        tensor_product = UnitaryDefinitions.sigmax().tensor(
            UnitaryDefinitions.sigmax())

        assert tensor_product.get_dimension() == dimension ** 2
        assert not tensor_product.close_to(np.identity(dimension ** 2))

        tensor_product = tensor_product.left_multiply(tensor_product)
        assert tensor_product.close_to(np.identity(dimension ** 2))

    def test_multiply(self):
        dimension = 2
        identity = Unitary.identity(dimension)

        product = Unitary(dimension)
        product = product.left_multiply(UnitaryDefinitions.sigmax())
        assert product.close_to(UnitaryDefinitions.sigmax())

        product = product.right_multiply(UnitaryDefinitions.sigmay())
        assert product.close_to(UnitaryDefinitions.sigmaz())

        product = product.left_multiply(UnitaryDefinitions.sigmaz())
        assert product.close_to(identity)

    def test_rphi(self):
        theta_values = [0, np.pi/8, np.pi/4, np.pi, 3*np.pi/2, -np.pi/4]
        for theta in theta_values:
            assert UnitaryDefinitions.rphi(theta, 0).close_to(
                UnitaryDefinitions.rx(theta))
            assert UnitaryDefinitions.rphi(theta, np.pi/2).close_to(
                UnitaryDefinitions.ry(theta))

    def test_display_name(self):
        dimension = 2
        operation_name = "Rx"
        unitary = Unitary(dimension, np.array([
            [np.exp(1j*np.pi/4), 0],
            [0, 1j]]), operation_name)
        display_name_with_zero_parameters = unitary.get_display_name()
        assert isinstance(display_name_with_zero_parameters, str)
        assert operation_name == display_name_with_zero_parameters

        parameter_name_1 = "abc"
        unitary = Unitary(
            dimension, unitary.get_matrix(), operation_name,
            {parameter_name_1: (1.0, True)})
        display_name_with_one_parameter = unitary.get_display_name()
        assert isinstance(display_name_with_one_parameter, str)
        assert operation_name in display_name_with_one_parameter

        parameter_name_2 = "def"
        unitary = Unitary(
            dimension, unitary.get_matrix(), operation_name,
            {parameter_name_1: (1.0, True), parameter_name_2: (2.0, False)})
        display_name_with_two_parameters = unitary.get_display_name()
        print(display_name_with_two_parameters)
        assert isinstance(display_name_with_two_parameters, str)
        assert operation_name in display_name_with_two_parameters
        assert parameter_name_1 in display_name_with_two_parameters
        assert parameter_name_2 in display_name_with_two_parameters

    def test_qasm(self):
        dimension = 2
        operation_name = "Rx"
        unitary = Unitary(dimension, np.array([
            [np.exp(1j*np.pi/4), 0],
            [0, 1j]]), operation_name)
        assert unitary.get_qasm() == operation_name + "\t" + "q[0];"

    def test_gms(self):
        for num_qubits in [3, 4, 5]:
            u = UnitaryDefinitions.gms(num_qubits)
            assert (u.left_multiply(u).left_multiply(u).left_multiply(u)
                    .close_to(Unitary.identity(u.get_dimension())))


class TestParameterizedUnitary:

    def test_parameterized_rotation(self):
        dimension = 2

        def rotation_matrix(alpha, beta, gamma):
            return np.array(
                [[np.cos(beta/2) * np.exp(-1j*(alpha+gamma)/2),
                    -np.sin(beta/2) * np.exp(-1j*(alpha-gamma)/2)],
                 [np.sin(beta/2) * np.exp(1j*(alpha-gamma)/2),
                    np.cos(beta/2) * np.exp(1j*(alpha+gamma)/2)]])

        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter(
                        "alpha", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter(
                        "beta", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter(
                        "gamma", min_value, max_value, is_angle=True)]
        operation_name = "R"

        rotation = ParameterizedUnitary(
            dimension, rotation_matrix, parameters, operation_name)

        zero_rotation_unitary = rotation.as_unitary([0, 0, 0])
        assert zero_rotation_unitary.close_to(Unitary.identity(dimension))
        assert operation_name in zero_rotation_unitary.get_display_name()
        assert [
            p.get_parameter_name()
            in zero_rotation_unitary.get_display_name()
            for p in parameters]

        random_values = [p.random_value() for p in parameters]
        assert np.all([
            parameters[i].is_valid(r)
            for i, r in enumerate(random_values)])

        random_rotation_unitary = rotation.as_unitary(random_values)
        assert operation_name in random_rotation_unitary.get_display_name()
        assert [
            p.get_parameter_name()
            in random_rotation_unitary.get_display_name()
            for p in parameters]
        assert random_rotation_unitary.left_multiply(
            random_rotation_unitary.inverse()).close_to(
                Unitary.identity(dimension))

    def test_parameterized_unitary_classmethods(self):
        rotation_xy = ParameterizedUnitaryDefinitions.rotation_xy()
        zero_rotation_unitary = rotation_xy.as_unitary([0, 0])
        assert zero_rotation_unitary.close_to(
            Unitary.identity(rotation_xy.get_dimension()))

        rotation_xyz = ParameterizedUnitaryDefinitions.rotation_xyz()
        zero_rotation_unitary = rotation_xyz.as_unitary([0, 0, 0])
        assert zero_rotation_unitary.close_to(
            Unitary.identity(rotation_xyz.get_dimension()))

        xx = ParameterizedUnitaryDefinitions.xx()
        xx_angle = 2*np.pi
        full_rotation_unitary = xx.as_unitary([xx_angle])
        assert full_rotation_unitary.close_to(
            Unitary.identity(xx.get_dimension()))

        angle_parameter_name = xx.get_parameters()[0].get_parameter_name()
        assert (xx_angle, True) == full_rotation_unitary.get_parameter_value(
            angle_parameter_name)

    def test_parameterized_unitary_time_evolution(self):
        sigmax = np.array([[0, 1], [1, 0]])
        t_min = -1.234
        t_max = 1.234
        time_evolution = ParameterizedUnitaryDefinitions.time_evolution(
            sigmax, t_min, t_max)
        zero_time_unitary = time_evolution.as_unitary([0])
        assert zero_time_unitary.close_to(
            Unitary.identity(time_evolution.get_dimension()))

        evolution_time = t_max / 2.0
        forward_time_unitary = time_evolution.as_unitary([evolution_time])
        backward_time_unitary = time_evolution.as_unitary([-evolution_time])
        assert forward_time_unitary.close_to(backward_time_unitary.inverse())

        time_parameter_name = (time_evolution.get_parameters()[0]
                               .get_parameter_name())
        assert ((evolution_time, False) ==
                forward_time_unitary.get_parameter_value(time_parameter_name))
        assert ((-evolution_time, False) ==
                backward_time_unitary.get_parameter_value(time_parameter_name))

        with pytest.raises(Exception):
            # switch ordering of t_min and t_max
            time_evolution = ParameterizedUnitaryDefinitions.time_evolution(
                sigmax, t_max, t_min)

        with pytest.raises(Exception):
            # time outside valid range
            time_evolution.as_unitary([2 * t_max])


class TestUnitarySequenceEntry:

    def test_identity(self):
        dimension = 2
        entry = UnitarySequenceEntry(Unitary.identity(dimension), [0])
        assert entry.get_dimension() == dimension
        assert np.array_equal(entry.get_apply_to(), [0])

        for system_dimension in [2, 4, 8, 16]:
            full_unitary = entry.get_full_unitary(system_dimension)
            assert full_unitary.close_to(Unitary.identity(system_dimension))

    def test_cnot(self):
        entry = UnitarySequenceEntry(UnitaryDefinitions.cnot(), [0, 1])

        with pytest.raises(Exception):
            system_dimension = 2
            full_unitary = entry.get_full_unitary(system_dimension)

        system_dimension = 4
        full_unitary = entry.get_full_unitary(system_dimension)
        assert full_unitary.close_to(UnitaryDefinitions.cnot())

        system_dimension = 8
        full_unitary = entry.get_full_unitary(system_dimension)
        assert full_unitary.close_to(
            UnitaryDefinitions.cnot().tensor(Unitary.identity(2)))

        system_dimension = 16
        full_unitary = entry.get_full_unitary(system_dimension)
        assert full_unitary.close_to(
            UnitaryDefinitions.cnot().tensor(Unitary.identity(4)))
        assert full_unitary.left_multiply(full_unitary).close_to(
            Unitary.identity(system_dimension))

    def test_cnot_swapped(self):
        dimension = 4
        entry = UnitarySequenceEntry(UnitaryDefinitions.cnot(), [1, 0])

        system_dimension = 4
        full_unitary = entry.get_full_unitary(system_dimension)
        assert full_unitary.close_to(Unitary(dimension, np.array(
            [[1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]])))


class TestUnitarySequence:

    def test_default(self):
        dimension = 2
        sequence = UnitarySequence(dimension)

        assert sequence.get_dimension() == dimension
        assert sequence.product().close_to(np.identity(dimension))

    def test_identity_roots_correct(self):
        dimension = 2
        t = Unitary(dimension, np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]))
        t_entry = UnitarySequenceEntry(t, [0])
        sequence = UnitarySequence(dimension, np.repeat(t_entry, 8))

        assert sequence.get_dimension() == dimension
        assert sequence.get_length() == 8
        assert sequence.product().close_to(np.identity(dimension))

    def test_identity_roots_incorrect(self):
        dimension = 2
        t = Unitary(dimension, np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]))
        t_entry = UnitarySequenceEntry(t, [0])
        sequence = UnitarySequence(dimension, np.repeat(t_entry, 7))

        assert sequence.get_dimension() == dimension
        assert sequence.get_length() == 7
        assert not sequence.product().close_to(np.identity(dimension))

    def test_append_and_remove(self):
        dimension = 2
        identity = Unitary.identity(dimension)

        sequence = UnitarySequence(dimension)
        assert sequence.get_length() == 0
        assert sequence.product().close_to(identity)

        sequence.append_first(
            UnitarySequenceEntry(UnitaryDefinitions.sigmax(), [0]))
        assert sequence.get_length() == 1
        assert sequence.product().close_to(UnitaryDefinitions.sigmax())

        sequence.append_last(
            UnitarySequenceEntry(UnitaryDefinitions.sigmay(), [0]))
        assert sequence.get_length() == 2
        assert sequence.product().close_to(UnitaryDefinitions.sigmaz())

        sequence.append_first(
            UnitarySequenceEntry(UnitaryDefinitions.sigmaz(), [0]))
        assert sequence.get_length() == 3
        assert sequence.product().close_to(identity)

        sequence.remove_last()
        assert sequence.get_length() == 2
        assert sequence.product().close_to(UnitaryDefinitions.sigmay())

        sequence.remove_first()
        assert sequence.get_length() == 1
        assert sequence.product().close_to(UnitaryDefinitions.sigmax())

        sequence.remove_first()
        assert sequence.get_length() == 0
        assert sequence.product().close_to(identity)

    def test_undo(self):
        dimension = 2

        identity = Unitary.identity(dimension)

        sequence = UnitarySequence(dimension)
        assert sequence.get_length() == 0

        with pytest.raises(Exception):
            sequence.undo()

        sequence.append_first(
            UnitarySequenceEntry(UnitaryDefinitions.sigmax(), [0]))
        assert sequence.get_length() == 1
        assert sequence.product().close_to(UnitaryDefinitions.sigmax())

        sequence.undo()
        assert sequence.get_length() == 0
        assert sequence.product().close_to(identity)

        with pytest.raises(Exception):
            sequence.undo()

        sequence.append_first(
            UnitarySequenceEntry(UnitaryDefinitions.sigmay(), [0]))
        sequence.append_first(
            UnitarySequenceEntry(UnitaryDefinitions.sigmay(), [0]))
        assert sequence.get_length() == 2
        assert sequence.product().close_to(identity)

        sequence.remove_last()
        assert sequence.get_length() == 1
        assert sequence.product().close_to(UnitaryDefinitions.sigmay())

        sequence.undo()
        assert sequence.get_length() == 2
        assert sequence.product().close_to(identity)

        with pytest.raises(Exception):
            sequence.undo()

    def test_combine(self):
        dimension = 2
        t = Unitary(dimension, np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]))
        t_entry = UnitarySequenceEntry(t, [0])
        sequence_1 = UnitarySequence(
            dimension, np.repeat(t_entry, 3))
        sequence_2 = UnitarySequence(
            dimension, [
                UnitarySequenceEntry(UnitaryDefinitions.sigmay(), [0])])

        combined_sequence = UnitarySequence.combine(sequence_1, sequence_2)
        assert (combined_sequence.get_length() ==
                (sequence_1.get_length() + sequence_2.get_length()))
        assert combined_sequence.product().close_to(
            sequence_1.product().left_multiply(sequence_2.product()))

    def test_inverse(self):
        dimension = 2
        rx_entry = UnitarySequenceEntry(UnitaryDefinitions.rx(np.pi/3), [0])
        ry_entry = UnitarySequenceEntry(UnitaryDefinitions.ry(np.pi/3), [0])
        sequence = UnitarySequence(dimension, [rx_entry, ry_entry])
        product = sequence.product()

        inverse_sequence = sequence.inverse()
        inverse_product = inverse_sequence.product()
        assert inverse_product.close_to(product.inverse())

        inverse_sequence.sequence_product = None
        inverse_product = inverse_sequence.product()
        assert inverse_product.close_to(product.inverse())

    def test_qasm(self):
        dimension = 2
        rx_entry = UnitarySequenceEntry(UnitaryDefinitions.rx(np.pi/3), [0])
        ry_entry = UnitarySequenceEntry(UnitaryDefinitions.ry(np.pi/3), [0])
        sequence = UnitarySequence(dimension, [rx_entry, ry_entry])
        assert sequence.get_qasm()
        assert sequence.get_display_output()
