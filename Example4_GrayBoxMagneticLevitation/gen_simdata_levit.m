% Simone Pirrera
% % Oct 15, 2024
% Simulated data generator for magnetic levitation system

% This script generates synthetic data for a magnetic levitation using the
% corresponding gray-box model.
% Due to the system's instability, a stabilizing controller is designed to
% run the data collection simulation.
% The parameters of mass, initial displacement, and magnetic constants are
% taken according to the values estimated through ad-hok expeiments on
% "Levitator Sigma" at LADISPE Laboratory at Politecnico di Torino.

clear; close all; clc;

%% Parameters: standard estimation
% Parameters of Sigma device.
m = 24.197e-3;     % directly measured
g = 9.81;          % constant
z0 = 0.0335;       % directly measured
% First estimation: position-voltage transducer
DATA1 = readmatrix("misurazioni_Sigma\posizione_sigma");
z1 = DATA1(:,1); v1 = DATA1(:,2);
figure, plot(z1,v1,'b');  grid on;
xlabel("$z$ (m)", 'Interpreter','latex','FontSize',22);
ylabel("$V_o$ (V)", 'Interpreter','latex','FontSize',22);
idx_start = 35; idx_end = 85;
hold on; plot(z1(idx_start:idx_end),v1(idx_start:idx_end),'g'); 
transducer_params = [z1(idx_start:idx_end),ones(idx_end-idx_start+1,1)]\v1(idx_start:idx_end);
Kt = transducer_params(1);
v0 = transducer_params(2);
hold on; plot(z1(idx_start:idx_end),Kt*z1(idx_start:idx_end)+v0,'r'); 
clear DATA1;

% Second estimation: position-voltage transducer
% data at equilibrium: mg = F = (Km i^2 + k0)/z^2
DATA1 = readmatrix("misurazioni_Sigma\magnete_sigma");
z2 = DATA1(:,2); i2 = DATA1(:,3);
dz = z2-z0;
figure, plot(dz,i2,'b'); grid on;
magnet_params = [i2.^2]\(m*g*dz.^2);
Km = magnet_params(1);
k0 = 0; %magnet_params(2);
hold on; plot(dz,sqrt((m*g*dz.^2-k0)/Km),'r'); 
clear DATA1;

%% Linearized plant and stabilizing controller
% State x = [z; \dot z]
ibar = sqrt((m*g*(mean(z1(idx_start:idx_end))-z0).^2-k0)/Km);
x = sym('x',[2,1],'real');
u = sym('u','real');
f = [x(2); g-(Km/m)*(u^2/x(1)^2)];
x_eq = solve(subs(f,u,ibar),x);
z_bar = double(x_eq.x1(2));

A = double(subs(jacobian(f,x), [x;u], [z_bar,0,ibar]'));
B = double(subs(jacobian(f,u), [x;u], [z_bar,0,ibar]'));
C = [1 0];

pcl = 5*[-1 -1.1];
K = place(A,B,pcl);

%% Simulation, plot, and save data
ref = z_bar;
epsilon = 0.001;
zini = z_bar + 2*epsilon*rand - epsilon;

Ts = 1e-2;  Tsim = 100;
out = sim("levitat_model.slx");
figure, plot(out.z.time, out.z.data); grid on;
figure, plot(out.i_in.time, out.i_in.data, 'g'); grid on;
figure, plot(out.vz.time, out.vz.data, 'k'); grid on;

save levitat_data.mat u y Ts Kt v0 ibar Km k0 m z_bar g;