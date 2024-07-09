from scipy.linalg import expm
from scipy import sparse
import numpy as np
import scipy.io as spio
import scipy
import pickle
from bqskit.compiler import Compiler

from bqskit.compiler import Workflow
from bqskit import Circuit


from bqskit.passes import LEAPSynthesisPass,QFASTDecompositionPass, QuickPartitioner, QSearchSynthesisPass



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

loadMat = spio.loadmat('./alanineNMRdata_withrelaxation.mat',squeeze_me=True)

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

#defining custom workflow...
#inst_opts = {'method':'qfactor'} we can define the a QFactor-instantiation-based workflow, but it fails when using it 
#as an option for QSearch and QFast

custom_workflow = Workflow([
    QuickPartitioner(3),
    QSearchSynthesisPass(success_threshold=1e-3), 
    #QSearchSynthesisPass(success_threshold=1e-3, instantiate_options=inst_opts) 
    ])


circ = Circuit.from_unitary(USub)

with Compiler() as compiler:
    opt_circuit = compiler.compile(circ, workflow=custom_workflow)

Dict = {'circuit': opt_circuit}

with open('customcirc_t'+str(1)+'.pk', 'wb') as handle:
    pickle.dump(Dict, handle)

print("Depth of optimized circuit: ", opt_circuit.depth)