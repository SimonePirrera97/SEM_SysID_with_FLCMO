% Training of NNARX and NNOE using toolbox
% Simone Pirrera
% Dec 2, 2023
% Traning of SISO and MIMO NNOE networks for system identification 

clear; close all; clc;

PATH_DATA = ".\\data";
% Import dataset
load(strcat(PATH_DATA,"\\data_es2.mat"));

utrain = u_train; ytrain = y_train;
N_train = size(utrain,1);
yt = ytrain(1:N_train,:); ut = utrain(1:N_train,:);
N = size(yt,1); 

%% Define network
NetDef = ['HHHHHH';'L-----'];

n = 4;
NN=[n n+1 0];
trparms = settrain;
trparms = settrain(trparms,'maxiter',1e5,'critterm',1e-12,'gradterm',1e-12);

NUM_RUNS = 10;
for run = 1:NUM_RUNS
    tic
    [W1,W2,critvec,iter,lambda]=nnoe(NetDef,NN,[],[],trparms,yt',ut');
    time_nnsyd(run) = toc;
    NNOE_Net.W{1} = W1(:,1:end-1);
    NNOE_Net.W{2} = W2(:,1:end-1);
    NNOE_Net.B{1} = W1(:,end);
    NNOE_Net.B{2} = W2(:,end);
    NNOE_Net.n = n; NNOE_Net.dim_y = 1;
    NNOE_Net.N_layers = 1;
     
    % Validation
    Nvalid = length(u_valid);
    yp = zeros(Nvalid,1);
    for k = n+1:Nvalid
        r = [y_valid(k-1:-1:k-n);u_valid(k:-1:k-n)];
        yp(k) = NNOE_Net.W{2}*tanh(NNOE_Net.W{1}*r+NNOE_Net.B{1})+NNOE_Net.B{2};
    end
    y_sim_test = simulateMultiLayerNNOE(NNOE_Net,u_valid,y_valid(1:2));
    
    MSE_nnsyd(run) = (1/(2*Nvalid))*sum((y_sim_test(:,1)-y_valid(:,1)).^2);
    FIT_nnsyd(run) = 100*(1-norm(y_sim_test-y_valid)/norm(y_valid-mean(y_valid)));
    fprintf(1, 'MSE test: %e\n', MSE_nnsyd(run))
    fprintf(1, 'FIT test: %e\n', FIT_nnsyd(run))
    
end
save nnsyd_results_6neur_order4.mat MSE_nnsyd FIT_nnsyd time_nnsyd;


fprintf(1, 'FIT test mean (std): %.2f (%.2f)\n', mean(FIT_nnsyd), std(FIT_nnsyd));
fprintf(1, 'MSE test mean (std): %.2f (%.2f)\n', mean(MSE_nnsyd), std(MSE_nnsyd));
fprintf(1, 'training time mean (std): %.2f (%.2f)\n', mean(time_nnsyd), std(time_nnsyd));