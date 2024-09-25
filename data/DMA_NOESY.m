clc
clear all

props = gparse('dimethylacetamide.log');
[sys,inter]=g2spinach(props,{{'H','1H'}},31.8);

zeeman_avg1 = (inter.zeeman.matrix{1} + inter.zeeman.matrix{2} + inter.zeeman.matrix{3})/3;
zeeman_avg2 = (inter.zeeman.matrix{4} + inter.zeeman.matrix{5} + inter.zeeman.matrix{6})/3;
zeeman_avg3 = (inter.zeeman.matrix{7} + inter.zeeman.matrix{8} + inter.zeeman.matrix{9})/3;

inter.zeeman.matrix{1} = zeeman_avg1;
inter.zeeman.matrix{2} = zeeman_avg1;
inter.zeeman.matrix{3} = zeeman_avg1;

inter.zeeman.matrix{4} = zeeman_avg2;
inter.zeeman.matrix{5} = zeeman_avg2;
inter.zeeman.matrix{6} = zeeman_avg2;

inter.zeeman.matrix{7} = zeeman_avg3;
inter.zeeman.matrix{8} = zeeman_avg3;
inter.zeeman.matrix{9} = zeeman_avg3;

J = inter.coupling.scalar;
Ja = (J{1,2}+J{2,3}+J{1,3})/3;
Jb = (J{4,5}+J{5,6}+J{4,6})/3;
Jc = (J{7,8}+J{8,9}+J{7,9})/3;
Jab = (J{1,4}+J{1,5}+J{1,6}+J{2,4}+J{2,5}+J{2,6}+J{3,4}+J{3,5}+J{3,6})/9;
Jac = (J{1,7}+J{1,8}+J{1,9}+J{2,7}+J{2,8}+J{2,9}+J{3,7}+J{3,8}+J{3,9})/9;
Jbc = (J{4,7}+J{5,8}+J{6,9}+J{4,7}+J{5,8}+J{6,9}+J{4,7}+J{5,8}+J{6,9})/9;

J{1,2} = Ja;
J{2,3} = Ja;
J{1,3} = Ja;

J{4,5} = Jb;
J{5,6} = Jb;
J{4,6} = Jb;

J{7,8} = Jc;
J{8,9} = Jc;
J{7,9} = Jc;

[J{1,4:6}] = deal(Jab);
[J{2,4:6}] = deal(Jab);
[J{3,4:6}] = deal(Jab);

[J{1,7:9}] = deal(Jac);
[J{2,7:9}] = deal(Jac);
[J{3,7:9}] = deal(Jac);

[J{4,7:9}] = deal(Jbc);
[J{5,7:9}] = deal(Jbc);
[J{6,7:9}] = deal(Jbc);

for i = 1:9
    for j = 1:9
        J{j,i} = J{i,j};
    end
end
inter.coupling.scalar = J;


sys.magnet=14.1;
bas.formalism='sphten-liouv';
% bas.approximation='none';
bas.approximation='IK-2';
bas.connectivity='scalar_couplings';
bas.space_level=3;
tols.prox_cutoff = 100;
bas.sym_group={'S3','S3','S3'};
bas.sym_spins={[1 2 3],[4 5 6],[7 8 9]};
% Tolerances
sys.tols.inter_cutoff=1;
sys.enable={'greedy'};



inter.relaxation={'redfield'};
inter.equilibrium='zero';
inter.temperature=310;
inter.rlx_keep='secular';
inter.tau_c={0.05e-9};

spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

parameters.tmix=0.5;
parameters.offset= 000;
parameters.sweep=[5000 5000];
parameters.npoints=[1024 1024];
parameters.zerofill=4*[1024 1024];
parameters.spins={'1H'};
parameters.axis_units='ppm';
parameters.rho0=state(spin_system,'Lz','1H','chem');

[fid,H,R,K]=liquid1(spin_system,@noesy1,parameters,'nmr');
% [fid]=liquid(spin_system,@noesy,parameters,'nmr');

%% Perform matrix exponentiation checks 
% (note: using @liquid1 for H, R, K and @noesy1 which does not have magnetic field gradients)

dt=1./parameters.sweep;
Lx=operator(spin_system,'Lx',parameters.spins{1});
Ly=operator(spin_system,'Ly',parameters.spins{1});
coil=state(spin_system,'L+',parameters.spins{1},'cheap');

% first 90x pulse
U90x = expm(-1i*Lx*pi/2);
rho_initial = U90x*parameters.rho0;
dim = length(parameters.rho0);
% t1 evolutionl
L_net=H+1i*R+1i*K;
L_dt1 = expm(-1i*L_net*dt(1));
rho_stack = zeros(dim,parameters.npoints(1));
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

rho_stack1 = zeros(dim,parameters.npoints(1),4);
Mat1 = pulse_90y*pulse_mix*pulse_90x;
Mat2 = pulse_90y*pulse_mix*pulse_90y;
Mat3 = pulse_90y*pulse_mix*pulse_90mx;
Mat4 = pulse_90y*pulse_mix*pulse_90my;
for i = 1:parameters.npoints(1)
    rho_stack1(:,i,1) = Mat1*rho_stack(:,i);
    rho_stack1(:,i,2) = Mat2*rho_stack(:,i);
    rho_stack1(:,i,3) = Mat3*rho_stack(:,i);
    rho_stack1(:,i,4) = Mat4*rho_stack(:,i);
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

%% Plot

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
        20,[0.01 0.1 0.01 0.1],2,256,6,'both');

%% Save data

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
p.spin_system = spin_system;

save DMA.mat p
























% spin_system=create(sys,inter);
% spin_system=basis(spin_system,bas);

% % Sequence parameters
% parameters.spins={'1H'};
% parameters.rho0=state(spin_system,'L+','1H','cheap');
% parameters.coil=state(spin_system,'L+','1H','cheap');
% parameters.decouple={};
% parameters.offset=2000;
% parameters.sweep=6000;
% parameters.npoints=2^12;
% parameters.zerofill=2^13;
% parameters.axis_units='ppm';
% parameters.invert_axis=1;
% 
% % Simulation
% fid=liquid(spin_system,@acquire,parameters,'nmr');
% 
% % Apodization
% fid=apodization(fid,'gaussian-1d',10);
% 
% % Fourier transform
% spectrum=fftshift(fft(fid,parameters.zerofill));
% 
% % Plotting
% figure(); plot_1d(spin_system,real(spectrum),parameters);
% 
% return











