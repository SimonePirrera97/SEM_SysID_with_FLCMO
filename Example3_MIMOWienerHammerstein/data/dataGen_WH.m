% Simone Pirrera
% Data generation for Polynomial WH MIMO system.

clear; close all; clc; 

Ts = 1e-2; Nsim = 50;
myseed = [10; 54];
out = sim("PolyMIMOsystem.slx");

utrain = [out.u.data];
ytrain = [out.y.data];

figure, plot(utrain);
figure, plot(ytrain); legend('$y_1$', '$y_2$','Interpreter', 'latex', 'FontSize',22)

Nsim = 50;
myseed = [7; 27];
out = sim("PolyMIMOsystem.slx");
u_test = [out.u.data];
y_test = [out.y.data];
figure, plot(y_test); legend('$y_1$', '$y_2$','Interpreter', 'latex', 'FontSize',22)

save dataset_training.mat ytrain utrain
save dataset_test.mat y_test u_test 