import os
from timeit import default_timer as timer
import pickle
import numpy as np
from scipy.linalg import expm
import scipy
import sys

from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit import MachineModel
from bqskit.ir.gates import CZGate, RZGate, SXGate
from bqskit import compile
from cirq.contrib.qasm_import import circuit_from_qasm
import cirq

import logging
from bqskit.compiler import Compiler
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import QFASTDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import ToU3Pass
from bqskit.passes import ToVariablePass
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import UnfoldPass
from bqskit.passes import SimpleLayerGenerator
from bqskit.ir.gates import VariableUnitaryGate
from bqskit import Circuit


def loadMat(mol,path):
    fname=path+f'generators_noesy_{mol}.mat'

    return spio.loadmat(fname, squeeze_me=True)

def EmbedInU(TarMat):

    Dim = TarMat.shape[0]

    UR = scipy.linalg.sqrtm(np.eye(Dim)-np.dot(TarMat,TarMat.conjugate().T))
    LL = scipy.linalg.sqrtm(np.eye(Dim)-np.dot(TarMat.conjugate().T,TarMat))
    
    U_meth = np.zeros([2*Dim,2*Dim],dtype=complex)
    U_meth[0:Dim,0:Dim] = TarMat
    U_meth[0:Dim,Dim:2*Dim]=UR
    U_meth[Dim:2*Dim,0:Dim]=LL
    U_meth[Dim:2*Dim,Dim:2*Dim]=-TarMat.conjugate().T

    return U_meth

def Umetric(TarMat):
    dim = TarMat.shape[0]
    
    return np.linalg.norm(np.dot(TarMat.conj().T,TarMat)-np.eye(dim))


def run_simp_layer_flow_example(in_circuit,
        amount_of_workers: int = 10, synt_pass = QSearchSynthesisPass
) -> tuple[Circuit, float]:
    
    num_multistarts = 32
   
    instantiate_options = {
        'method': 'qfactor',
        'multistarts': num_multistarts,
    }

    passes = [

        # Split the circuit into partitions
       #QSearchSynthesisPass(instantiate_options=instantiate_options),
       synt_pass(layer_generator=SimpleLayerGenerator(two_qudit_gate=VariableUnitaryGate(2),single_qudit_gate_1=VariableUnitaryGate(1)),
                 success_threshold=1e-3,max_layer=5000,instantiate_options=instantiate_options)
       
       #QSearchSynthesisPass(layer_generator=LayerGenDef.AltLayer(),instantiate_options=instantiate_options)

    ]
    

    with Compiler(
        num_workers=amount_of_workers,
        runtime_log_level=logging.INFO,
    ) as compiler:

        print('Starting flow using QFactor instantiation')
        start = timer()
        out_circuit = compiler.compile(in_circuit, passes)
        end = timer()
        run_time = end - start

    return out_circuit, run_time




def SimulateBlock(bqskit_circ,n_flag,reps=1000,gate_set={CZGate(), RZGate(), SXGate()},noise=None):
    """
    Simulate post-selected samples from a Block-encoding unitary using cirq. Assuming that the target n-qubit matrix is block encoded in the 
    upper-left block of the unitary, this corresponds to post-select the measurement outcomes in the last n-qubits of the circuit.
    Args:
    bqskit_circ: result of circuit synthesis
    n_flag: number of flag qubits for the block encoding
    gate_set: the target gate set to perform the compilation
    """

    model = MachineModel(bqskit_circ.num_qudits, gate_set=gate_set)

    inst_circuit = compile(bqskit_circ, model=model)
    lang = OPENQASM2Language()
    qasm = lang.encode(inst_circuit)

    cirq_circ = circuit_from_qasm(qasm)
    qubits = sorted(cirq_circ.all_qubits())
    
    Nqubs = len(cirq_circ.all_qubits())

    n_sys = Nqubs - n_flag
    #qubits = cirq.LineQubit.range(Nqubs)
    control_register = qubits[0:n_flag]
    target_register = qubits[n_flag:]
    
    
    cirq_circ.append(cirq.measure(*control_register, key='control')) 
    cirq_circ.append(cirq.measure(*target_register, key='target'))
    
    
    simulator = cirq.Simulator()
    result = simulator.run(cirq_circ, repetitions=reps)
    
    # Step 4: Post-select results based on control register measurements
    control_measurements = result.measurements['control']
    target_measurements = result.measurements['target']
    
    # Post-select where control register is [0, 0] (or any desired condition)
    #post_selected_indices = np.where((control_measurements[:, 0] == 0) & (control_measurements[:, 1] == 0))[0]
    post_selected_indices = np.where((control_measurements[:] == [0]*n_flag))[0]
    post_selected_target_measurements = target_measurements[post_selected_indices]

    return post_selected_target_measurements

def EstimatePolarization(Measurements):
    """
    Estimate the expectation value of S_{z} = \sum_{n}\sigma^{(z)}_{n} from a list of Measurements
    """
    nqubs = len(Measurements[0])

    Tot_pol=0.0
    for i in range(len(Measurements)):

        m = 0.0
        for j in range(nqubs):
            res=Measurements[i][j]
            m+=(-1.0)**res

        Tot_pol+=m

    return Tot_pol/len(Measurements)


