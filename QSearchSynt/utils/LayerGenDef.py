#Customizing layer generator for compilation....
from __future__ import annotations

import logging

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.ry import RYGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates import VariableUnitaryGate

from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
_logger = logging.getLogger(__name__)



class AltLayer(LayerGenerator):
    """
    A customized alternate-layer generator class.

    It introduces a (VariableUnitaryGate) single-qubit layer in the initial step


    This is an optimized layer generator that uses commutativity rules
    to reduce the number of parameters per block. This also fixes the
    gate set to use cnots, ry, rz, and u3 gates. This is based on the
    following equivalences:

    U--C--U   U--C--Rz--Ry--Rz   U--C--Ry--Rz
       |    ~    |             ~    |
    U--X--U   U--X--Rx--Ry--Rx   U--X--Ry--Rx
    """



    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` is not qubit only.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if not target.is_qubit_only():
            raise ValueError('Cannot generate layers for non-qubit circuits.')

        init_circuit = Circuit(target.num_qudits, target.radixes)
        for i in range(init_circuit.num_qudits):
            init_circuit.append_gate(VariableUnitaryGate(1), [i])
        return init_circuit



    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            ValueError: If circuit is a single-qudit circuit.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if circuit.num_qudits < 2:
            raise ValueError('Cannot expand a single-qudit circuit.')

        # Get the machine model
        #coupling_graph = data.connectivity

        # Generate successors
        successors = []

        #Layering depends on the parity of the layer we are constructing...
        last = circuit.last_on(1)[0]

        parity = last%2

        #We construct the coupling graph...
        coupling_graph = self.build_coup_graph(parity,circuit.num_qudits)
        print(coupling_graph)

        for edge in coupling_graph:

            #if self.count_outer_cnots(circuit, edge) >= 3:
                # No need to build circuits with more than 3 cnots in a row
            #    if circuit.num_qudits != 2:
                    # Guard on >2 qubit to prevent high-error glitches
            #        continue

            successor = circuit.copy()

            successor.append_gate(VariableUnitaryGate(2), edge)
            #successor.append_gate(RYGate(), edge[0])
            #successor.append_gate(RZGate(), edge[0])
            #successor.append_gate(RYGate(), edge[1])
            #successor.append_gate(RXGate(), edge[1])
            successors.append(successor)

        return successors

    def build_coup_graph(self,parity,nqubs):
        
        graph = []

        counter=0
        for i in range(nqubs//2):
            if (parity+2*counter+1)<nqubs:
                graph.append((parity+2*counter,parity+2*counter+1))
            counter+=1

        return graph



    def count_outer_cnots(self, circuit: Circuit, edge: tuple[int, int]) -> int:
        """
        Count how many uninterrupted 4-param cnot blocks are on `edge`.

        This will count backwards from the right-side of the circuit and stop
        when a cnot is encountered including other qudits.
        """
        rear_point = circuit._rear[edge[0]]
        num_cx_seen = 0

        if rear_point is None or rear_point.cycle != circuit.num_cycles - 1:
            return num_cx_seen

        while rear_point is not None:
            rear_points = circuit.prev(rear_point)

            if len(rear_points) == 0:
                break

            rear_point = rear_points.pop()

            if circuit[rear_point].num_qudits == 1:
                # Move past single-qubit gates
                continue

            cx_op = circuit[rear_point]

            if cx_op.location != edge:
                # If CX is on a different edge stop counting
                break

            num_cx_seen += 1
