import tequila as tq
import numpy as np
import openfermion
import scipy

#yogo
path_prefix="../CSA-master/"
import sys
sys.path.append(path_prefix)
sys.path.append("./")

from tequila.grouping import binary_rep as BinRep
from tequila.grouping import binary_utils as BinUtils

#We can straightforwardly obtain the maximally abelian subgroup of all the possible Pauli products
#by mapping the Pauli words that comprise the qubit Hamiltonian to the so-called binary matrix (which is obtained
#by 1) calling to function init_from_qubit_hamiltonian(H), H being the Hamiltonian in a qubit encoding; and 2) using the
#get_binary function of the resulting object) and computing the lagrangian subspace of the latter.
#The arrows of the last matrix can be mapped back to their Pauli product form by changing the data type of the
#arrow vector to BinaryPauliString and calling to_pauli_strings() function from the resulting object.

#For future reference, the binary vectors returned by the nullspace generator must be multiplied by the simpletic
#matrix to map them to the Pauli words that commute with the Hamiltonian

from tequila.hamiltonian import QubitHamiltonian
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.binary_utils import get_lagrangian_subspace
from tequila.grouping.binary_rep import BinaryPauliString
import tequila.grouping.binary_rep as BinRep


def get_one_body_terms(H):
    '''
    Return the one body terms in H
    '''
    one_body = openfermion.FermionOperator()
    for fw, val in H.terms.items():
        if len(fw) <= 2:
            one_body += openfermion.FermionOperator(fw, val)
    return one_body

def get_two_body_terms(H):
    '''
    Return the one body terms in H
    '''
    two_body = openfermion.FermionOperator()
    for fw, val in H.terms.items():
        if len(fw) > 2:
            two_body += openfermion.FermionOperator(fw, val)
    return two_body


def GenQubitSym(qubHam,verif=True):
    '''
    Function that returns the maximal abelian subgroup of Pauli words that commute with the Hamiltonian.
    Input: mol, string for the name of the molecule of interest; path_prefix, path to CSA-master, that contains
    the electronic Hamiltonian data.
    '''
     #Qubit Hamiltonian in tequila qubit operator class
    tq_qubHam=QubitHamiltonian.from_openfermion(qubHam)
    BinMat=BinaryHamiltonian.init_from_qubit_hamiltonian(tq_qubHam)
    BinMat=BinMat.get_binary()

    LagMat=get_lagrangian_subspace(BinMat)

    nPaul=np.shape(LagMat)[0]

    suma=openfermion.QubitOperator()
    ArrayOps=[]
    for i in range(nPaul):
        #suma+=BinaryPauliString(LagMat[i]).to_pauli_strings()
        PaulW=BinaryPauliString(LagMat[i]).to_pauli_strings()
        Dum=openfermion.QubitOperator(PaulW.key_openfermion())
        suma+=Dum
        ArrayOps.append(Dum)
        #print(PaulW.to_openfermion())
        #print(PaulW)
        #print(openfermion.QubitOperator(PaulW.key_openfermion()))

    #we can verify the results
    if verif:
        AbQub=openfermion.QubitOperator()
        for i in suma.get_operators():
            for j in suma.get_operators():
                AbQub+=openfermion.utils.commutator(i,j)

        ComHam=openfermion.utils.commutator(qubHam,suma)

        return ArrayOps,AbQub,ComHam
    else:
        return ArrayOps

def GenCliffUnit(hqub):
    '''
    Function that returns the Clifford unitary (as an OpenFermion qubit operator object) that renders a qubit Hamiltonian
    into "qubit-tapperable" form.
    Input:hqub, an openfermion Hamiltonian operator;
    '''

    tq_qubHam=QubitHamiltonian.from_openfermion(hqub)
    BinMat=BinaryHamiltonian.init_from_qubit_hamiltonian(tq_qubHam)

    LagMat=get_lagrangian_subspace(BinMat.get_binary())

    dim = len(LagMat)

    # Free Qubits
    #free_qub = [qub for qub in range(dim)]
    free_qub = [qub for qub in range(len(LagMat[0])//2)]
    pair = []

    for i in range(dim):
        #while cur_pair is None:
            #print("Entered")
        cur_pair = BinMat.find_single_qubit_pair(LagMat[i],
                                               free_qub)
        if cur_pair is None:
            while cur_pair is None:
                cur_pair = BinMat.find_single_qubit_pair(LagMat[i],
                                               free_qub)


        #print(cur_pair)
        for j in range(dim):
            if i != j and cur_pair is not None:
                #print("Entered here")
                if BinRep.binary_symplectic_inner_product(
                        cur_pair, LagMat[j]==1):
                    print("Entered here")
                    LagMat[j] = (LagMat[i] +
                                           LagMat[j]) % 2
        pair.append(cur_pair)

    #print(len(pair))
    #CliffUnit=openfermion.QubitOperator()
    CliffUnit=1.0
    for i in range(len(LagMat)):
        Unit1=openfermion.QubitOperator()
        Dum1=BinaryPauliString(LagMat[i]).to_pauli_strings()
        Dum2=BinaryPauliString(pair[i]).to_pauli_strings()
                               # BinaryPauliString(pair[0]).to_pauli_strings())

        Dum1=openfermion.QubitOperator(Dum1.key_openfermion())
        Dum2=openfermion.QubitOperator(Dum2.key_openfermion())

        Unit1+=(1.0/np.sqrt(2))*Dum1+(1.0/np.sqrt(2))*Dum2
        CliffUnit=CliffUnit*Unit1

    return CliffUnit

def ReduceQubs(TappHam,TapIdxs,nqubs):
    '''
    Function that re-labels the indexes of the Pauli words of a tapered-qubit operator. This is introduced
    in order to make openfermion to recognize a lower number of qubits. For instance,
    if we consider a 10 qubit operator whose 2nd and 4th are tapered-off, openfermion would still
    recognize it as a 10-qubit operator, as the labels of the tapered operator are not contigous.
    Input: TapIdxs, list that contains the indexes of the tapered qubits
    '''
    #TapIdxs=[0,15]
    #Build dictionary...
    OrigIdxs=[dum for dum in range(nqubs)]

    for i in range(len(TapIdxs)):
        OrigIdxs.remove(TapIdxs[i])

    TransIdxs={}

    for i in range(len(OrigIdxs)):
        TransIdxs[OrigIdxs[i]]=i

    #print(TransIdxs[2])

    QubTap=openfermion.QubitOperator()
    for i in TappHam.terms:
        if np.abs(TappHam.terms[i]) >=1e-5 and i!=():
            dum=TappHam.terms[i]
            for j in range(len(i)):
                dum=dum*openfermion.QubitOperator(i[j][1]+str(TransIdxs[i[j][0]]))
            QubTap+=dum

    QubTap+=TappHam.constant
    return QubTap


def TapperRotHam(EigNums,RotSyms,RotHam,nqubs):
    '''
    Function to tapper-off a qubit Hamiltonian. It is assumed that the latter is already
    in the basis where the qubit symmetries are single qubit observables.
    Input: RotSyms, list of the single-qubit symmetries that are going to be tappered-off;
    RotHam, the qubit Hamiltonian in the same basis as RotSyms, where the qubits
    are tappered-off; EigNums, list whose ith element corresponds to the eigenvalue of
    the ith symmetry stored in RotSyms; nqubs is the number of qubits that span the total Hilbert space

    '''
    TapHam=RotHam
    counter=0
    count_aps=0
    for k in RotSyms: #iterate over the qubits to tapper off

        for i in TapHam.get_operators(): #iterate over all the contributions of the rotated Hamiltonian
            New=i

            for j in i.terms: #Iterate over the Pauli operators of each Pauli word

                for l in range(len(j)):

                    if openfermion.QubitOperator(j[l])==k:
                        #print("Got here")
                        count_aps+=1
                        TapHam=TapHam-New
                        New=New*k*EigNums[counter]
                        TapHam=TapHam+New

        #TapHam+=New
        counter+=1

    #Notice that here we make use of the assumption that RotSyms is a list of single
    #qubit operators only..
    TapIdxs=[]
    for i in range(len(RotSyms)):
        for j in RotSyms[i].terms:
            TapIdxs.append(j[0][0])
    #print(TapIdxs)

    #nqubs=openfermion.count_qubits()

    #ReduceQubs(TapHam,TapIdxs,nqubs)

    return ReduceQubs(TapHam,TapIdxs,nqubs)


def EigSymGS(vec,nqubs,QubOp):
    '''
    Auxiliary function to compute the eigenvalue of the qubit symmetry operators on a given state vector.
    Input: vec, an eigenvector of a given Hamiltonian; nqubs, number of qubits used for the encoding;
    QubOp, an openfermion Qubit operator that corresponds to one of the qubit symmetries found with our methods.

    '''

    SpOp=openfermion.get_sparse_operator(QubOp,n_qubits=nqubs)

    return np.dot(np.conjugate(vec[:,0]),SpOp*vec[:,0])



def SwapOp(p,q):

    return 1-openfermion.FermionOperator(str(p)+'^ '+str(p))-openfermion.FermionOperator(str(q)+'^ '+str(q))+openfermion.FermionOperator(str(p)+'^ '+str(q))+openfermion.FermionOperator(str(q)+'^ '+str(p))
