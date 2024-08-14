clc
clear all

% Alpha outside
sys.isotopes={'19F','19F','19F','19F'};
inter.zeeman.scalar={-120.8361 -134.2763 -129.5988 -129.7528};   
inter.coupling.scalar=cell(4,4);
inter.coupling.scalar{1,2}=  271.2924;
inter.coupling.scalar{3,4}=  271.2924;
inter.coupling.scalar{1,3}=    0.5401;
inter.coupling.scalar{1,4}=  -25.9884;
inter.coupling.scalar{2,3}=    9.9625;
inter.coupling.scalar{2,4}=  -40.7675;        
inter.coordinates={[-0.0551   -1.2087   -1.6523];
                   [-0.8604   -2.3200   -0.0624];
                   [-2.4464   -0.1125   -0.9776];
                   [-1.9914   -0.0836    1.0743]};
sys.magnet=9.3933;

bas.formalism='sphten-liouv';
bas.approximation='none';

inter.relaxation={'redfield'};
inter.equilibrium='dibari';
inter.temperature=310;
inter.rlx_keep='secular';
inter.tau_c={0.951e-9};

spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

parameters.tmix=0.5;
parameters.offset=-49000;
parameters.sweep=[8e3 8e3];
parameters.npoints=[256 256];
parameters.zerofill=[1024 512];
parameters.spins={'19F'};
parameters.axis_units='ppm';
% parameters.needs={'rho_eq'};
parameters.rho0=state(spin_system,'Lz','19F','chem');

[fid,H,R,K]=liquid1(spin_system,@noesy1,parameters,'nmr');

%% Perform matrix exponentiation checks 
% (note: using @liquid1 for H, R, K and @noesy1 which does not have magnetic field gradients)

dt=1./parameters.sweep;
Lx=operator(spin_system,'Lx',parameters.spins{1});
Ly=operator(spin_system,'Ly',parameters.spins{1});
coil=state(spin_system,'L+',parameters.spins{1},'cheap');

% % % L_net=H+1i*R+1i*K;
% % % L_dt1 = expm(-1i*L_net*dt(1));
% % % L_dt2 = expm(-1i*L_net*dt(2));
% % % U_mix = expm(-1i*L_net*parameters.tmix);
% % % U90x = expm(-1i*Lx*pi/2);
% % % U90y = expm(-1i*Ly*pi/2);
% % % rho1 = L_dt2*L_dt2*U90y*U_mix*U90y*L_dt1*U90x*parameters.rho0;
% % % rho2 = L_dt2*L_dt2*U90y*U_mix*(U90y')*L_dt1*U90x*parameters.rho0;
% % % 
% % % a1 = trace(coil'*rho1) - trace(coil'*rho2)
% % % a2 = fid.sin(3,2)
% % % abs(a1-a2)

% first 90x pulse
U90x = expm(-1i*Lx*pi/2);
rho_initial = U90x*parameters.rho0;

% t1 evolutionl
L_net=H+1i*R+1i*K;
L_dt1 = expm(-1i*L_net*dt(1));
rho_stack = zeros(256,parameters.npoints(1));
rho_stack(:,1) = rho_initial;
rho_temp = rho_initial;
for i=2:parameters.npoints(1)
   rho_temp=L_dt1*rho_temp;
   rho_stack(:,i) = rho_temp;
end

% second 90 deg pulse (90x, 90y, -90x, -90y), followed by mixing, and the
% third 90 deg pulse (90y)
pulse_90x = expm(-1i*Lx*pi/2);
pulse_90y = expm(-1i*Ly*pi/2);
pulse_90mx = expm(1i*Lx*pi/2);
pulse_90my = expm(1i*Ly*pi/2);
pulse_mix = expm(-1i*L_net*parameters.tmix);

rho_stack1 = zeros(256,parameters.npoints(1),4);
for i = 1:parameters.npoints(1)
    rho_stack1(:,i,1) = pulse_90y*pulse_mix*pulse_90x*rho_stack(:,i);
    rho_stack1(:,i,2) = pulse_90y*pulse_mix*pulse_90y*rho_stack(:,i);
    rho_stack1(:,i,3) = pulse_90y*pulse_mix*pulse_90mx*rho_stack(:,i);
    rho_stack1(:,i,4) = pulse_90y*pulse_mix*pulse_90my*rho_stack(:,i);
end

% calculate fid
L_dt2 = expm(-1i*L_net*dt(2));
fid_temp = zeros(parameters.npoints(2),parameters.npoints(1),4);
for i = 1:parameters.npoints(1)
    rho1 = rho_stack1(:,i,1);
    rho2 = rho_stack1(:,i,2);
    rho3 = rho_stack1(:,i,3);
    rho4 = rho_stack1(:,i,4);    
    for j = 1:parameters.npoints(2)
        fid_temp(j,i,1) = trace(coil'*rho1);
        rho1 = L_dt2*rho1;

        fid_temp(j,i,2) = trace(coil'*rho2);
        rho2 = L_dt2*rho2;

        fid_temp(j,i,3) = trace(coil'*rho3);
        rho3 = L_dt2*rho3;

        fid_temp(j,i,4) = trace(coil'*rho4);
        rho4 = L_dt2*rho4;

    end
end

fid_test.cos = fid_temp(:,:,1) - fid_temp(:,:,3);
fid_test.sin = fid_temp(:,:,2) - fid_temp(:,:,4);
% fid.cos = fid_temp(:,:,1) - fid_temp(:,:,3);
% fid.sin = fid_temp(:,:,2) - fid_temp(:,:,4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Apodization
fid.cos=apodization(fid.cos,'sqcosbell-2d');
fid.sin=apodization(fid.sin,'sqcosbell-2d');

% F2 Fourier transform
f1_cos=real(fftshift(fft(fid.cos,parameters.zerofill(2),1),1));
f1_sin=real(fftshift(fft(fid.sin,parameters.zerofill(2),1),1));

% States signal
f1_states=f1_cos-1i*f1_sin;

% F1 Fourier transform
spectrum=fftshift(fft(f1_states,parameters.zerofill(1),2),2);

% Plotting
figure(); scale_figure([1.5 2.0]);
plot_2d(spin_system,-real(spectrum),parameters,...
        20,[0.01 0.2 0.01 0.2],2,256,6,'positive');



%% save parameters
dt=1./parameters.sweep;
time_grid1 = 0:dt(1):(parameters.npoints(1)-1)*dt(1); % t1 evolution
time_grid2 = 0:dt(2):(parameters.npoints(2)-1)*dt(2); % final detection


p = parameters;
p.time_grid1 = time_grid1;
p.time_grid2 = time_grid2;
p.H = H;
p.R = R;
p.fid = fid;
p.fid_test = fid_test; % the fid's obtained via explicit matrix exponentiation
p.R = R;

save TFG.mat p

