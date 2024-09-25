clc
clear all

% props=gparse('/usr/software/spinach/examples/standard_systems/alanine.log');
props=gparse('alanine.log');
[sys,inter]=g2spinach(props,{{'H','1H'}},31.8);
H_index = [2 3 4 5];

sys.isotopes = {'1H','1H','1H','1H'};
coordinates = cell(4,1);
for i = 1:4
    coordinates{i} = inter.coordinates{H_index(i)};
end
inter.coordinates = coordinates;

zeeman = cell(1,4);
for i = 1:4
    zeeman{i} = inter.zeeman.matrix{H_index(i)};
end
inter.zeeman.matrix = zeeman;
zeeman_1H_avg = (zeeman{2} + zeeman{3} + zeeman{4})/3;
inter.zeeman.matrix{2} = zeeman_1H_avg;
inter.zeeman.matrix{3} = zeeman_1H_avg;
inter.zeeman.matrix{4} = zeeman_1H_avg;

coupling = cell(4);
for i = 1:4
    for j = 1:4
        coupling{i,j} = inter.coupling.scalar{H_index(i),H_index(j)};
    end
end
inter.coupling.scalar = coupling;
JM = (coupling{2,3} + coupling{2,4} + coupling{3,4})/3;
JMH = (coupling{1,2} + coupling{1,3} + coupling{1,4})/3;
[inter.coupling.scalar{1,2:4}] = deal(JMH);
inter.coupling.scalar{2,3} = JM;
inter.coupling.scalar{2,4} = JM;
inter.coupling.scalar{3,4} = JM;
for i = 1:4
    for j = 1:4
        inter.coupling.scalar{j,i} = inter.coupling.scalar{i,j};
    end
end

% Magnet field
sys.magnet=14.1;

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='none';

bas.sym_group={'S3'};
bas.sym_spins={[2 3 4]};

% Relaxation theory parameters
inter.relaxation={'redfield'};
inter.equilibrium='zero';
inter.temperature=310;
inter.rlx_keep='secular';
inter.tau_c={0.05e-9};

% inter.equilibrium='dibari';
% inter.temperature=300;
% inter.relaxation={'t1_t2'};
% inter.r1_rates={2 2 2 2};
% inter.r2_rates={5 5 5 5};
% inter.rlx_keep='secular';

% Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

% Sequence parameters
parameters.tmix=1;
parameters.offset= 1500;
parameters.sweep=[4000 4000];
parameters.npoints=[1024 1024];
parameters.zerofill=4*[1024 1024];
parameters.spins={'1H'};
parameters.axis_units='ppm';
parameters.rho0=state(spin_system,'Lz','1H','chem');

[fid,H,R,K]=liquid1(spin_system,@noesy1,parameters,'nmr');
% [fid]=liquid(spin_system,@noesy,parameters,'nmr');


% %% Perform matrix exponentiation checks 
% % (note: using @liquid1 for H, R, K and @noesy1 which does not have magnetic field gradients)
% 
% dt=1./parameters.sweep;
% Lx=operator(spin_system,'Lx',parameters.spins{1});
% Ly=operator(spin_system,'Ly',parameters.spins{1});
% coil=state(spin_system,'L+',parameters.spins{1},'cheap');
% 
% % first 90x pulse
% U90x = expm(-1i*Lx*pi/2);
% rho_initial = U90x*parameters.rho0;
% dim = length(parameters.rho0);
% % t1 evolutionl
% L_net=H+1i*R+1i*K;
% L_dt1 = expm(-1i*L_net*dt(1));
% rho_stack = zeros(dim,parameters.npoints(1));
% rho_stack(:,1) = rho_initial;
% rho_temp = rho_initial;
% 
% for i=2:parameters.npoints(1)
%    rho_temp=L_dt1*rho_temp;
%    rho_stack(:,i) = rho_temp;
% end
% 
% % second 90 deg pulse (90x, 90y, -90x, -90y), followed by mixing, and the
% % third 90 deg pulse (90y)
% pulse_90x = expm(-1i*Lx*pi/2);
% pulse_90y = expm(-1i*Ly*pi/2);
% pulse_90mx = expm(1i*Lx*pi/2);
% pulse_90my = expm(1i*Ly*pi/2);
% pulse_mix = expm(-1i*L_net*parameters.tmix);
% 
% rho_stack1 = zeros(dim,parameters.npoints(1),4);
% Mat1 = pulse_90y*pulse_mix*pulse_90x;
% Mat2 = pulse_90y*pulse_mix*pulse_90y;
% Mat3 = pulse_90y*pulse_mix*pulse_90mx;
% Mat4 = pulse_90y*pulse_mix*pulse_90my;
% for i = 1:parameters.npoints(1)
%     rho_stack1(:,i,1) = Mat1*rho_stack(:,i);
%     rho_stack1(:,i,2) = Mat2*rho_stack(:,i);
%     rho_stack1(:,i,3) = Mat3*rho_stack(:,i);
%     rho_stack1(:,i,4) = Mat4*rho_stack(:,i);
% end
% 
% % calculate fid
% L_dt2 = expm(-1i*L_net*dt(2));
% fid_temp = zeros(parameters.npoints(2),parameters.npoints(1),4);
% for i = 1:parameters.npoints(1)
%     rho1 = rho_stack1(:,i,1);
%     rho2 = rho_stack1(:,i,2);
%     rho3 = rho_stack1(:,i,3);
%     rho4 = rho_stack1(:,i,4);    
%     for j = 1:parameters.npoints(2)
%         fid_temp(j,i,1) = trace(coil'*rho1);
%         rho1 = L_dt2*rho1;
% 
%         fid_temp(j,i,2) = trace(coil'*rho2);
%         rho2 = L_dt2*rho2;
% 
%         fid_temp(j,i,3) = trace(coil'*rho3);
%         rho3 = L_dt2*rho3;
% 
%         fid_temp(j,i,4) = trace(coil'*rho4);
%         rho4 = L_dt2*rho4;
% 
%     end
% end
% 
% fid_test.cos = fid_temp(:,:,1) - fid_temp(:,:,3);
% fid_test.sin = fid_temp(:,:,2) - fid_temp(:,:,4);

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
        20,[0.02 0.2 0.02 0.2],2,256,6,'both');












dt=1./parameters.sweep;
time_grid1 = 0:dt(1):(parameters.npoints(1)-1)*dt(1); % t1 evolution
time_grid2 = 0:dt(2):(parameters.npoints(2)-1)*dt(2); % final detection


p = parameters;
p.time_grid1 = time_grid1;
p.time_grid2 = time_grid2;
p.H = H;
p.R = R;
p.fid = fid;
% p.fid_test = fid_test; % the fid's obtained via explicit matrix exponentiation
p.R = R;

save NOESYdata_ALA_withGradients.mat p
