% SISO NNOE training.
% Simone Pirrera
% Dec 2, 2023
% Traning of NNOE networks for system identification 

clear; close all; clc;
PATH_DATA = ".\\data";

print_to_file = true;
if print_to_file
    fileID = fopen("res_cmo3.txt","w");
else
    fileID = 1;
end

% Import dataset
load(strcat(PATH_DATA,"\\data_es2.mat"));

utrain = u_train; ytrain = y_train;
N_train = size(utrain,1);
ytrain = {ytrain(1:N_train,:)}; utrain = {utrain(1:N_train,:)};
N = size(ytrain{1},1); 

% Define network
n = 4; [Nn,N_layers] = structureNNOE('6');
dim_u = size(utrain{1},2); dim_y = size(ytrain{1},2);
fprintf(1,"number of parameters: %d\n", Nn*(2*n+2)+Nn+1);

%% Training process
NUM_RUNS = 10;
for run = 1:NUM_RUNS
    % Initialization of the optimization variables
    [Net, numParams] = initializeMultiLayerNNOE(n, Nn, dim_u, dim_y);
    numOptimVars = numParams + dim_y*N;
    numConstraints = dim_y*(N-n);
    Net.numParams = numParams;
    
    numExperiments = length(ytrain);

    % Algorithm parameters
    param.Ts = 2e-3;
    param.max_iter = 1000;
    param.K = 1;
    param.rho = 1e-3; % Regularization cofficient
    param.display_skip = 100;
    param.tol = 1e-3;
    param.ask_continue = param.max_iter+1;
    param.printFile = fileID;
    param.method = "QR";

    fprintf(fileID,"  Run #%d\n",run);
    fprintf(1,"  Run #%d\n",run);
    % Initial conditions
    x0 = 1e-4*randn(numOptimVars,1);
    
    % RUN
    tic
    [Net,sol,iter] = train_nnoe(Net,utrain,ytrain,[],param);
    time_cmo(run) = toc;
    fprintf(1, 'time: %.2f\n', time_cmo(run));

    % Plots (uncomment to see plots)
    idx_out = 1;
    % figure, plot(1:iter, sol(1:numParams,1:iter));
    % title('Evolution of parameters estimate', 'FontSize',20);
    
    % Simulation on training data: check overfitting
    y_sim_train = simulateMultiLayerNNOE(Net, utrain{1}, ytrain{1}(1:Net.n));
    fprintf(1, 'RMSE training: %.2f\n', norm(y_sim_train(:,1)-ytrain{1}(:,1)))

    FIT_train = 100*(1-norm(y_sim_train-ytrain{1}(:,1))/norm(ytrain{1}(:,1)-mean(ytrain{1}(:,1))));
    fprintf(1, 'FIT training: %.2f\n', FIT_train)
    
    % figure, plot(ytrain{1}), hold on; plot(y_sim_train);
    % legend('true training output','simulated training output','FontSize',20);
    % title('Training data','FontSize',20);
    
    % Simulation on validation data: check model's quality
    N0_valid = 1;  Nvalid = length(y_valid);
    y0 = y_valid(N0_valid:Net.n+N0_valid-1,idx_out);
    utest = u_valid(N0_valid:end,:); 
    ytest = y_valid(N0_valid:end,idx_out);
    y_sim_test = simulateMultiLayerNNOE(Net, utest, y0);

    % figure, plot(ytest), hold on; plot(y_sim_test);
    % legend('true test output 1','simulated test output 1', 'FontSize',20);
    % title('Validation','FontSize',20);    
    MSE_cmo(run) = (1/(2*Nvalid))*sum((y_sim_test(:,1)-ytest(:,1)).^2);
    FIT_cmo(run) = 100*(1-norm(y_sim_test-y_valid)/norm(y_valid-mean(y_valid)));
    fprintf(1, 'MSE test: %.2f\n', MSE_cmo(run))
    fprintf(1, 'FIT test: %.2f\n', FIT_cmo(run))
    
end

if print_to_file
    fclose(fileID);
end

save cmo_results_6neur_order4_p2.mat MSE_cmo FIT_cmo time_cmo;

%% Load results and compute mean
fprintf(1, 'FIT test mean (std): %.2f (%.2f)\n', mean(FIT_cmo), std(FIT_cmo));
fprintf(1, 'MSE test mean (std): %.2f (%.2f)\n', mean(MSE_cmo), std(MSE_cmo));
fprintf(1, 'training time mean (std): %.2f (%.2f)\n', mean(time_cmo), std(time_cmo));