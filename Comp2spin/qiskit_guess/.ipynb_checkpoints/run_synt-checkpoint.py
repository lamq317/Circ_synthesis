from __future__ import annotations

import logging
import os
from timeit import default_timer as timer
import pickle
import numpy as np
from scipy.linalg import expm
import scipy
import sys
sys.path.append('./utils/')
#import LayerGenDef

#from LayerGenDef import AltLayer


from bqskit import Circuit
from bqskit import enable_logging
from bqskit.compiler import Compiler
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import QFASTDecompositionPass
from bqskit.passes import UnfoldPass
from bqskit.passes import QuickPartitioner

import qiskit
from bqskit.ext import qiskit_to_bqskit


#from qfactorjax.qfactor import QFactorJax
#from qfactorjax.qfactor_sample_jax import QFactorSampleJax

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
        amount_of_workers: int = 10, partition_size = 6
) -> tuple[Circuit, float]:

    inst_opts = {'method':'minimization'}

    passes = [
            # Convert U3's to VU
            #ToVariablePass(convert_all_single_qudit_gates=True),

            # Split the circuit into partitions
            QuickPartitioner(partition_size),

            # For each partition perform scanning gate removal using QFactor jax
            ForEachBlockPass([
                ScanningGateRemovalPass(
                    instantiate_options=inst_opts,
                ),
            ]),

            # Combine the partitions back into a circuit
            UnfoldPass(),

            # Convert back the VariablueUnitaires into U3s
            #ToU3Pass(),
        ]


    with Compiler(
        num_workers=amount_of_workers,
        runtime_log_level=logging.INFO,
    ) as compiler:

        print('Starting flow using GateRemoval pass')
        start = timer()
        out_circuit = compiler.compile(in_circuit, passes)
        end = timer()
        run_time = end - start

    return out_circuit, run_time



if __name__ == '__main__':

    synt_pass = 'qsearch' if len(sys.argv) < 2 else sys.argv[1]
    nworkers = 6 if len(sys.argv) < 3 else int(sys.argv[2])
    part_size = 6 if len(sys.argv) < 4 else int(sys.argv[3])
    t_idx = 256 if len(sys.argv) < 5 else int(sys.argv[4])


    loadMat = spio.loadmat('../../data/DFG.mat',squeeze_me=True)

    Ham = loadMat['p']['H'].item()
    R = loadMat['p']['R'].item()
    t_grid = loadMat['p']['time_grid1'].item()

    TimeGen = (-1j*Ham+R)*t_grid[t_idx]
    ExpGen = expm(TimeGen)

    ####Embedding in unitary...

    EmbUn = EmbedInU(ExpGen)

    QisCircQs = qiskit.synthesis.qs_decomposition(EmbUn)
    bqskit_circuit = qiskit_to_bqskit(QisCircQs)

    #in_circuit = Circuit.from_unitary(EmbUn)

    out_circuit, run_time = run_gate_del_flow_example(bqskit_circuit,amount_of_workers= nworkers, partition_size = part_size)

    Dict = {'circuit': out_circuit}

    filename = './outputs/'+'sampqf_'+synt_pass+'t_'+str(t_idx)+str(nworkers)+'.pk'

    with open(filename,'wb') as handle:
        pickle.dump(Dict, handle)

    print(
        f'Synthesis took {run_time}'
        f'seconds using Sample QFactor JAX instantiation method.',
    )

    print(
        f'Circuit finished with gates: {out_circuit.gate_counts}, '
        f'while started with {bqskit_circuit.gate_counts}',
    )
