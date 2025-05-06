% MIMO NNOE training.
% Simone Pirrera
% Dec 2, 2023
% Traning of SISO and MIMO NNOE networks for system identification 

clear; close all; clc;

PATH_DATA = ".\\data"; 
% Import dataset: polynomial WH MIMO system
load(strcat(PATH_DATA,"\\dataset_training.mat"));
load(strcat(PATH_DATA,"\\dataset_test.mat"));
% fprintf(1,'Signal-to-noise ratio: %f, %f\n', SNR1, SNR2);
idx_out = [1:2];  % select both output signals
ytrain = ytrain(:,idx_out);

N_train = 3500; 
yt = ytrain(1:N_train,:); ut = utrain(1:N_train,:);
N = size(yt,1); 

% Define network
n = 3; [Nn,N_layers] = structureNNOE('8 8');
dim_u = size(ut,2); dim_y = size(yt,2);

%% Data pre-processing
sigma = ones(dim_y,1); % disabled scaling
for idx = 1:dim_y
    yt(:,idx) = (yt(:,idx))./sigma(idx);
end
figure, plot(yt)

%% Controlled Multipliers Optimization: Initilizations
% Initialization of the optimization variables
[Net, numParams] = initializeMultiLayerNNOE(n, Nn, dim_u, dim_y);
fprintf(1,"number of parameters: %d\n", numParams);
numOptimVars = numParams + dim_y*N;
numConstraints = dim_y*(N-n);

% Define FL-CMO gain
K = 100;

% Regularization cofficient
rho = 1e-4;
fprintf('rho = %e\n', rho);

% Initial conditions
x0 = 0.01*randn(numOptimVars,1);

%% Controlled Multipliers Optimization: Run
Ts = 0.01;  Nsim = 3000;
% static controller
sol = zeros(numOptimVars, Nsim); sol(:,1) = x0;
norm_h_val_vec = zeros(Nsim,1); best_train_error = 1e10;

tol = 1e-4;
norm_sol_dot = 1e10; norm_h_val = 1e10;
iter = 1;
tic
while max(norm_sol_dot,norm_h_val)>tol && iter<Nsim
    [sol_dot, h_val, Net0] = closedloop_odefun_qr(sol(:,iter),Net,ut,yt,[K;rho]);
    
    norm_sol_dot = norm(sol_dot);
    norm_h_val = norm(h_val);
    norm_h_val_vec(iter) = norm_h_val;

    sol(:,iter+1) = sol(:,iter) + Ts*sol_dot; % -> Euler

    iter = iter+1;
    if mod(iter,20)==0
        fprintf(1,"iter %d: ||h(x)||=%e, ||x_dot||=%e, ", iter, norm_h_val, norm_sol_dot );
        %Net0 = multiLayerNetFromVec(sol(:,i), n, Nn, dim_u, dim_y);
        y_sim_train = simulateMultiLayerNNOE(Net0, ut, yt(1:Net.n,:));
        train_error = norm(y_sim_train-yt);
        if train_error < best_train_error
            best_train_error_iter = iter;
            best_train_error = train_error;
            bestNet = Net0;
        end
        fprintf(1, 'RMSE: %e\n', train_error);
    end
end
req_time = toc;

fprintf(1, 'Steps: %d, Time: %f\n', iter, req_time);
fprintf(1, 'Best network at iteration: %d\n', best_train_error_iter);
%t = Ts*(0:Ts:Ts*(size(sol,2)))';

figure, plot(norm_h_val_vec)

%% Plots
figure, plot(1:iter, sol(1:numParams,1:iter));
title('Evolution of parameters estimate', 'FontSize',20);

Net = bestNet; 
%Net = multiLayerNetFromVec(sol(:,iter), n, Nn, dim_u, dim_y);

% Simulation on training data: check overfitting
y_sim_train = simulateMultiLayerNNOE(Net, ut, yt(1:Net.n));
if dim_y == 1
    fprintf(1, 'RMSE training: %e\n', norm(y_sim_train(:,1)-yt(:,1)))
    figure, plot(yt), hold on; plot(y_sim_train);
    legend('true training output 1','simulated training output 1','FontSize',20);
elseif dim_y == 2
    fprintf(1, 'RMSE training: %e, %e\n', norm(y_sim_train(:,1)-yt(:,1)), norm(y_sim_train(:,2)-yt(:,2)))
    figure, plot(yt), hold on; plot(y_sim_train);
    legend('true training output 1','true training output 2', ...
    'simulated training output 1','simulated training output 2', 'FontSize',20);
end

% Simulation on validation data: check model's quality
N0_valid = 1; y0 = y_test(N0_valid:Net.n+N0_valid-1,idx_out);
utest = u_test(N0_valid:end,:); ytest = y_test(N0_valid:end,idx_out);
y_sim_test = simulateMultiLayerNNOE(Net, utest, y0);
for idx = 1:dim_y
    y_sim_test(:,idx) = y_sim_test(:,idx).*sigma(idx);
end
if dim_y == 2
    figure, plot(ytest), hold on; plot(y_sim_test);
    legend('true test output 1','true test output 2', ...
    'simulated test output 1','simulated test output 2', 'FontSize',20);
    fprintf(1, 'RMSE test: %e, %e\n', norm(y_sim_test(:,1)-ytest(:,1)), norm(y_sim_test(:,2)-ytest(:,2)))
elseif dim_y==1
    figure, plot(ytest), hold on; plot(y_sim_test);
    legend('true test output 1','simulated test output 1', 'FontSize',20);
    fprintf(1, 'RMSE test: %e\n', norm(y_sim_test(:,1)-ytest(:,1)))
end

save results_cmo.mat Net K sigma best_train_error_iter Nsim Ts N_train iter 



