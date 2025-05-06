%% Identification of levitator gray-box parameters by LEAST SQUARES.
% Simone Pirrera
% Oct 16, 2024
clear; close all; clc;
load('levitat_data.mat');

N = 200;
rr = (m*y(1:N-2).^2).*( (y(3:N)-2*y(2:N-1)+y(1:N-2))/Ts^2 -g);
i2 = u(1:N-2).^2;
h = rr + Km*i2;

theta_hat = [i2, ones(N-2,1)]\(-rr);

figure, plot(h)

theta = [Km;k0];
disp(table(theta,theta_hat))