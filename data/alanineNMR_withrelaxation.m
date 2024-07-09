clc
clear all

sys.isotopes={'13C','13C','13C'};
sys.labels={'CB','CA','CO'};
sys.magnet=14.1;

inter.zeeman.scalar={18.83 53.2 178.56};
inter.coupling.scalar{1,2}=34.94;
inter.coupling.scalar{1,3}=1.21; 
inter.coupling.scalar{2,3}=53.81; 
inter.coupling.scalar{3,3}=0;

% inter.relaxation={'t1_t2'};
% inter.r1_rates={1/0.7 1/1.2 1/11.5};
% inter.r2_rates={1/0.81 1/0.41 1/1.3};

inter.rlx_keep='secular';
inter.relaxation={'lindblad'};
inter.lind_r1_rates=[1/0.7 1/1.2 1/11.5];
inter.lind_r2_rates=[1/0.81 1/0.41 1/1.3];
inter.equilibrium='zero'; 
inter.temperature=298;

bas.formalism='sphten-liouv';
bas.approximation='none';

spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);
spin_system=assume(spin_system,'nmr');

parameters.spins={'13C'};
parameters.rho0=state(spin_system,'L+','13C');
parameters.coil=state(spin_system,'L+','13C');
parameters.sweep=10000;
parameters.npoints=2^12;
parameters.zerofill=2^14;
parameters.axis_units='ppm';
parameters.invert_axis=1;

[fid,H,R]=liquid1(spin_system,@acquire,parameters,'nmr');

% check matrix exponential
dt = 1/parameters.sweep;
time_grid = 0:dt:(parameters.npoints-1)*dt;
U = eye(size(H));
Udt = expm(-1i*dt*(H+1i*R));
rho_now = parameters.rho0;
for i = 1:parameters.npoints
    fid_test(i,1) = trace(parameters.coil'*rho_now);
    rho_now = Udt*rho_now;
end

max(abs((abs(fid)-abs(fid_test))))

fid=apodization(fid,'gaussian-1d',10);
fid_test=apodization(fid_test,'gaussian-1d',10);

spectrum=fftshift(fft(fid,parameters.zerofill));
spectrum_test=fftshift(fft(fid_test,parameters.zerofill));

figure(); plot_1d(spin_system,real(spectrum),parameters);
hold on; plot_1d(spin_system,real(spectrum_test),parameters,'.','Color','k');

p = parameters;
p.H = H;
p.time_grid = time_grid;
p.fid = fid;
p.R = R;

save alanineNMRdata_withrelaxation.mat p




