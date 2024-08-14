#Assuming that we have access to a BQSkit circuit, we 1) transpile it to a CZ+single qubit rotation gate set.
#2) simulate the circuits (as a preliminary check, a noise-free simulation, and a simulation of the statistics of post-selection on the evolution)

import sys
sys.path.append('./utils/')

from simulation_utils import EmbedInU,run_simp_layer_flow_example, SimulateBlock,EstimatePolarization

#Auxiliary function to perform the apodization of FID
def sqcosbell_2d(shape):
    """
    Create a 2D squared cosine bell (SqCosBell) window.

    Parameters:
    shape (tuple): Shape of the window (height, width).

    Returns:
    np.ndarray: 2D SqCosBell window.
    """
    height, width = shape
    y = np.linspace(-np.pi, np.pi, height)
    x = np.linspace(-np.pi, np.pi, width)
    x, y = np.meshgrid(x, y)
    window = (np.cos(x / 2) ** 2) * (np.cos(y / 2) ** 2)
    return window


###Loading the generator of evolution...
loadMat = spio.loadmat('../data/DFG.mat',squeeze_me=True)

AuxMats = spio.loadmat('../data/DFG_NOESYmatrices.mat',squeeze_me=True)

Ham = loadMat['p']['H'].item()
R = loadMat['p']['R'].item()
t_grid1 = loadMat['p']['time_grid1'].item()
t_grid2 = loadMat['p']['time_grid2'].item()

###Dynamical evolution for calculation of 2D spectra...
##TODO: we can simply modify the script to incorporate 1) retrieval of synthesized circuits,
#and 2) the already-developed circuit simulator

Tpts = len(t_grid1)

Dim = Ham.shape[0]

rho0 = np.array(AuxMats['rho0'].toarray())
rho0 = rho0.flatten()
coil = AuxMats['coil']
#rho0 = np.zeros(Dim) #TODO: identify the basis convention to formulate the initial state
#rho0[2] = 0.5 #
#rho0[8] = 0.5

rho_t = np.copy(rho0)

tmix = 0.5

##Definition of pulses in the experiment...
#This also depends on the definition of the basis....
Lx = AuxMats['Lx'].toarray()
Ly = AuxMats['Ly'].toarray()

Lnet = -1j*Ham+R #TODO: In the matlab script we have an additional contribution K

pulse_90x = expm(-1j*Lx*np.pi/2)
pulse_90y = expm(-1j*Ly*np.pi/2)
pulse_90mx = expm(1j*Lx*np.pi/2)
pulse_90my = expm(1j*Ly*np.pi/2)
pulse_mix = expm(-1j*L_net*tmix)

FID_1 = np.array([Tpts,Tpts])
FID_2 = np.array([Tpts,Tpts])
FID_3 = np.array([Tpts,Tpts])
FID_4 = np.array([Tpts,Tpts])

#First 90x pulse:
rho_t = np.dot(pulse_90x,rho_t)

for i in range(Tpts):

    TimeGen = (-1j*Ham+R)*t_grid1[i]
    ExpGen = expm(TimeGen)
    rho_t = np.dot(ExpGen,rho_t)

    ###Perform 4 different experiments that correspond to 4 different signal sequences
    rho_t1 = pulse_90y@pulse_mix@pulse_90x@rho_t
    rho_t2 = pulse_90y@pulse_mix@pulse_90y@rho_t
    rho_t3 = pulse_90y@pulse_mix@pulse_90mx@rho_t
    rho_t4 = pulse_90y@pulse_mix@pulse_90my@rho_t
    #EmbUn = EmbedInU(ExpGen)
    for j in range(Tpts): #Assuming the number of points is the same as in t_grid2
        TimeGen = (-1j*Ham+R)*t_grid2[i]
        ExpGen = expm(TimeGen)

        FID_1[j,i] = np.dot(coil,rho_t1)
        rho_t1 = np.dot(ExpGen,rho_t1)

        FID_2[j,i] = np.dot(coil,rho_t2)
        rho_t2 = np.dot(ExpGen,rho_t2)

        FID_3[j,i] = np.dot(coil,rho_t3)
        rho_t3 = np.dot(ExpGen,rho_t3)

        FID_4[j,i] = np.dot(coil,rho_t4)
        rho_t4 = np.dot(ExpGen,rho_t4)

window = sqcosbell_2d(FID_1.shape)

FID_cos = FID_1 - FID_3
FID_sin = FID_2 -FID_4
##Apodization...
FID_cos = FID_cos*window
FID_sin = FID_sin*window

#2D Fourier transform...
f1_cos = np.real(np.fft.fftshift(np.fft.fft2(FID_cos)))
f1_sin = np.real(np.fft.fftshift(np.fft.fft2(FID_sin)))

f1_states = f1_cos-1j*f1_sin
spectrum = np.fft.fftshift(fft(f1_states)) #TODO: in the matlab
                                            #script, the Fourier transform
                                            #is carried out along the columnns
                                            #or rows, we need to double check this












