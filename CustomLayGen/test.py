from qfactorjax.qfactor import QFactorJax   
from qfactorjax.qfactor_sample_jax import QFactorSampleJax    
from bqskit.passes import SimpleLayerGenerator, FourParamGenerator
from timeit import default_timer as timer
import scipy.io as spio
from scipy.linalg import expm
import scipy
import numpy as np
from bqskit.ir.gates import VariableUnitaryGate

from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit import enable_logging
import logging

import sys
sys.path.append('./')
from LayerGenDef import AltLayer



from bqskit.passes import LEAPSynthesisPass

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


####Function to test the layer generator...
def TestLayGen(TarU,Lay_gen):
    num_multistarts = 10
    max_iters = 100000
    min_iters = 20
    diff_tol_r = 1e-5
    diff_tol_a = 0.0
    dist_tol = 1e-10

    diff_tol_step_r = 0.1
    diff_tol_step = 200

    beta = 0

    in_circuit = Circuit.from_unitary(TarU)

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
    #SimpleLayerGenerator(two_qudit_gate=CNOTGate, single_qudit_gate_1=U3Gate, single_qudit_gate_2=None, initial_layer_gate=None)

    passes = [
        # Convert U3's to VU
        #ToVariablePass(),

        # Split the circuit into partitions
       #QuickPartitioner(partition_size),
       #QSearchSynthesisPass(instantiate_options=instantiate_options),
       #LEAPSynthesisPass(layer_generator=SimpleLayerGenerator(two_qudit_gate=VariableUnitaryGate(2),
        #                                                      single_qudit_gate_1=VariableUnitaryGate(1)),
        #                                                     instantiate_options=instantiate_options)

        LEAPSynthesisPass(layer_generator=Lay_gen,instantiate_options=instantiate_options)

        # For each partition perform scanning gate removal using QFactor jax
        #ForEachBlockPass([
        #    ScanningGateRemovalPass(
        #        instantiate_options=instantiate_options,
        #    ),
        #]),

        # Combine the partitions back into a circuit
        #UnfoldPass(),

        # Convert back the VariablueUnitaires into U3s
        #ToU3Pass(),
    ]

    # Create the compilation task

    with Compiler(
        num_workers=1,
        runtime_log_level=logging.INFO,
    ) as compiler:

        print('Starting gate deletion flow using QFactor JAX')
        start = timer()
        out_circuit = compiler.compile(in_circuit, passes)
        end = timer()
        run_time = end - start

    return out_circuit, run_time


loadMat = spio.loadmat('../data/alanineNMRdata_withrelaxation.mat',squeeze_me=True)

Ham = loadMat['p']['H'].item()
R = loadMat['p']['R'].item()
t_grid = loadMat['p']['time_grid'].item()


TimeGen = (-1j*Ham+R)*t_grid[1]
ExpGen = expm(TimeGen)

####Embedding in unitary...

EmbUn = EmbedInU(ExpGen)

sub_qub = 3 #defines the size of the sub-matrix that is taken from the block encoding

SubMat = np.copy(EmbUn[0:2**sub_qub,0:2**sub_qub])

#Block encoding of the sub-matrix...
USub = EmbedInU(SubMat)

outcirc,runtime=TestLayGen(USub,AltLayer())

print(outcirc)
print("Running time is: ",runtime)