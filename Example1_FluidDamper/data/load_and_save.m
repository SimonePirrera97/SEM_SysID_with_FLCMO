clear; close all; clc;

load('mrdamper.mat');

% y = F
% u = V

z = [V,F];

N_train = 2000;

u_train = V(1:N_train); 
y_train = F(1:N_train);

u_valid = V(N_train+1:end); 
y_valid = F(N_train+1:end);

save data_es2.mat u_valid u_train y_train y_valid;


