#Running synthesis of 3 spin system...

from __future__ import annotations

import logging
import os
from timeit import default_timer as timer
import pickle
import numpy as np
from scipy.linalg import expm
import scipy
import qiskit


from bqskit import Circuit
from bqskit import enable_logging
from bqskit.compiler import Compiler
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QuickPartitioner
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import ToU3Pass
from bqskit.passes import ToVariablePass
from bqskit.passes import UnfoldPass
from bqskit.ext import qiskit_to_bqskit

from qfactorjax.qfactor import QFactorJax

import scipy.io as spio

enable_logging()

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


def run_gate_del_flow_example(in_circuit,
        amount_of_workers: int = 10,
) -> tuple[Circuit, float]:
    # The circuit to resynthesize
    #file_path = os.path.dirname(__file__) + '/grover5.qasm'

    # Set the size of partitions
    partition_size = 4

    # QFactor hyperparameters -
    # see instantiation example for more details on the parameters
    num_multistarts = 32
    max_iters = 100000
    min_iters = 3
    diff_tol_r = 1e-5
    diff_tol_a = 0.0
    dist_tol = 1e-10

    diff_tol_step_r = 0.1
    diff_tol_step = 200

    beta = 0

    #print(f'Will compile {file_path}')

    # Read the QASM circuit
    #in_circuit = Circuit.from_file(file_path)

    # Prepare the instantiator
    batched_instantiation = QFactorJax(
        diff_tol_r=diff_tol_r,
        diff_tol_a=diff_tol_a,
        min_iters=min_iters,
        max_iters=max_iters,
        dist_tol=dist_tol,
        diff_tol_step_r=diff_tol_step_r,
        diff_tol_step=diff_tol_step,
        beta=beta,
    )
    instantiate_options = {
        'method': batched_instantiation,
        'multistarts': num_multistarts,
    }

    # Prepare the compilation passes
    passes = [
        # Convert U3's to VU
        ToVariablePass(),

        # Split the circuit into partitions
        QuickPartitioner(partition_size),

        # For each partition perform scanning gate removal using QFactor jax
        ForEachBlockPass([
            ScanningGateRemovalPass(
                instantiate_options=instantiate_options,
            ),
        ]),

        # Combine the partitions back into a circuit
        UnfoldPass(),

        # Convert back the VariablueUnitaires into U3s
        ToU3Pass(),
    ]

    # Create the compilation task

    with Compiler(
        num_workers=amount_of_workers,
        runtime_log_level=logging.INFO,
    ) as compiler:

        print('Starting gate deletion flow using QFactor JAX')
        start = timer()
        out_circuit = compiler.compile(in_circuit, passes)
        end = timer()
        run_time = end - start

    return out_circuit, run_time


if __name__ == '__main__':

    loadMat = spio.loadmat('./data/alanineNMRdata_withrelaxation.mat',squeeze_me=True)

    Ham = loadMat['p']['H'].item()
    R = loadMat['p']['R'].item()
    t_grid = loadMat['p']['time_grid'].item()

    TimeGen = (-1j*Ham+R)*t_grid[1]
    ExpGen = expm(TimeGen)

    ####Embedding in unitary...

    EmbUn = EmbedInU(ExpGen)


    sub_qub = 4 #defines the size of the sub-matrix that is taken from the block encoding

    SubMat = np.copy(EmbUn[0:2**sub_qub,0:2**sub_qub])

    #Block encoding of the sub-matrix...
    USub = EmbedInU(SubMat)

    QisCircQs = qiskit.synthesis.qs_decomposition(USub)

    ###Transform to Circuit object 
    in_circuit = qiskit_to_bqskit(QisCircQs)

    #in_circuit = Circuit.from_unitary(USub)

    out_circuit, run_time = run_gate_del_flow_example(in_circuit)

    Dict = {'circuit': out_circuit}

    with open('testqis_jaxqfcirc_t'+str(1)+'.pk', 'wb') as handle:
        pickle.dump(Dict, handle)

    print(
        f'Partitioning + Synthesis took {run_time}'
        f'seconds using QFactor JAX instantiation method.',
    )

    print(
        f'Circuit finished with gates: {out_circuit.gate_counts}, '
        f'while started with {in_circuit.gate_counts}',
    )