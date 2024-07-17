#Running  complation of sub-matrix of block encoding of a 3-spin system using qfactor sample
#Usage: python samp_qf_synt_test.py [Method] [nworkers] [t_idx]
#current options [Method]=qsearch, leap


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
from bqskit.passes import ToU3Pass
from bqskit.passes import ToVariablePass
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import UnfoldPass
from bqskit.passes import SimpleLayerGenerator
from bqskit.ir.gates import VariableUnitaryGate

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
        amount_of_workers: int = 10, synt_pass = QSearchSynthesisPass
) -> tuple[Circuit, float]:
    
    num_multistarts = 32
    """

    qfactor_sample_gpu_instantiator = QFactorSampleJax(

    dist_tol=1e-8,       # Stopping criteria for distance

    max_iters=100000,      # Maximum number of iterations
    min_iters=6,          # Minimum number of iterations

    # Regularization parameter - [0.0 - 1.0]
    # Increase to overcome local minima at the price of longer compute
    beta=0.0,

    amount_of_validation_states=2,
    # indicates the ratio between the sum of parameters in the circuits to the
    # sample size.
    diff_tol_r=1e-4,
    num_params_coef=1,
    overtrain_relative_threshold=0.1,
    exact_amount_of_states_to_train_on=None,
    )

    """
    instantiate_options = {
        'method': 'qfactor',
        'multistarts': num_multistarts,
    }

    # Prepare the compilation passes
    #SimpleLayerGenerator(two_qudit_gate=CNOTGate, single_qudit_gate_1=U3Gate, single_qudit_gate_2=None, initial_layer_gate=None)

    passes = [

        # Split the circuit into partitions
       #QSearchSynthesisPass(instantiate_options=instantiate_options),
       synt_pass(layer_generator=SimpleLayerGenerator(two_qudit_gate=VariableUnitaryGate(2),single_qudit_gate_1=VariableUnitaryGate(1)),
                 success_threshold=1e-3,max_layer=5000,instantiate_options=instantiate_options)
       
       #QSearchSynthesisPass(layer_generator=LayerGenDef.AltLayer(),instantiate_options=instantiate_options)

    ]

    # Create the compilation task

    with Compiler(
        num_workers=amount_of_workers,
        runtime_log_level=logging.INFO,
    ) as compiler:

        print('Starting flow using Sample QFactor JAX')
        start = timer()
        out_circuit = compiler.compile(in_circuit, passes)
        end = timer()
        run_time = end - start

    return out_circuit, run_time


if __name__ == '__main__':

    synt_pass = 'qsearch' if len(sys.argv) < 2 else sys.argv[1]
    nworkers = 6 if len(sys.argv) < 3 else int(sys.argv[2])
    t_idx = 1 if len(sys.argv) < 4 else int(sys.argv[3])


    loadMat = spio.loadmat('../data/alanineNMRdata_withrelaxation.mat',squeeze_me=True)

    Ham = loadMat['p']['H'].item()
    R = loadMat['p']['R'].item()
    t_grid = loadMat['p']['time_grid'].item()

    TimeGen = (-1j*Ham+R)*t_grid[t_idx]
    ExpGen = expm(TimeGen)

    ####Embedding in unitary...

    EmbUn = EmbedInU(ExpGen)
    nqubs = 4
    SubU = np.copy(EmbUn[0:2**nqubs,0:2**nqubs])
    
    SubEmbU = EmbedInU(SubU)


    in_circuit = Circuit.from_unitary(SubEmbU)

    if synt_pass=='qsearch':
        out_circuit, run_time = run_gate_del_flow_example(in_circuit,amount_of_workers=nworkers,synt_pass=QSearchSynthesisPass)
    elif synt_pass=='leap':
        out_circuit, run_time = run_gate_del_flow_example(in_circuit,amount_of_workers=nworkers,synt_pass=LEAPSynthesisPass)
    else:
        print("Error, use either qsearch or leap for method")
        exit()

    Dict = {'circuit': out_circuit}

    filename = './outputs/'+'sampqf_'+synt_pass+'t_'+str(t_idx)+'_'+str(nworkers)+'.pk'

    with open(filename,'wb') as handle:
        pickle.dump(Dict, handle)

    print(
        f'Synthesis took {run_time}'
        f'seconds using Sample QFactor JAX instantiation method.',
    )

    print(
        f'Circuit finished with gates: {out_circuit.gate_counts}, '
        f'while started with {in_circuit.gate_counts}',
    )
