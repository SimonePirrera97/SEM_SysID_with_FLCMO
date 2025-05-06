function [bestNet,sol,iter] = train_nnoe(Net,ut,yt,x0,params)
%TRAIN_NNOE 

Ts = params.Ts;
max_iter = params.max_iter;
tol = params.tol;
K = params.K;
rho = params.rho;
numExperim = length(ut);
numOptimVars = Net.numParams;
problemInfo.method = params.method;

% Retrieve length of experiments and number of optimization variables
N = zeros(numExperim,1);
for idx = 1:numExperim
    N(idx) = size(ut{idx},1);
    numOptimVars = numOptimVars + Net.dim_y*N(idx);
end
problemInfo.N = N;

% Initializations
sol = zeros(numOptimVars, max_iter); 
if all(size(x0) == [numOptimVars,1])
    sol(:,1) = x0;
elseif numel(x0) == 0
    sol(:,1) = 0.01*randn(numOptimVars,1);
else
    fprintf(1,"Invalid initial conditions specified: x0 must have dimension [# optimization variables, 1] or be an empty vector x0=[].\n");
    fprintf(1,"Initializing x0 normally at random.\n");
end
    
norm_h_val_vec = zeros(max_iter,1); 
best_train_error = -1e10;
norm_sol_dot = 1e10; norm_h_val = 1e10; 

% Training loop
iter = 1;
tic
while max(norm_sol_dot,norm_h_val)>tol && iter<max_iter
    [sol_dot, h_val, Net0] = closedloop_odefun_qr_me(sol(:,iter),Net,ut,yt,[K;rho],problemInfo);
    
    norm_sol_dot = norm(sol_dot,inf);
    %norm_sol_dot_prec = norm(sol_dot,inf);
    norm_h_val = norm(h_val,inf);
    norm_h_val_vec(iter) = norm_h_val;

    % Euler integration
    sol(:,iter+1) = sol(:,iter) + Ts*sol_dot; 

    % Evaluate FIT error
    train_error = zeros(numExperim,1);
    rmse = zeros(numExperim,1);
    for ee = 1:numExperim
        y_sim_train = simulateMultiLayerNNOE(Net0, ut{ee}, yt{ee}(1:Net.n,:));
        train_error(ee) = 100*(1-norm(y_sim_train-yt{ee})/norm(yt{ee}-mean(yt{ee})));
        rmse(ee) = sqrt(1/N(ee))*norm(y_sim_train-yt{ee});
    end

    train_error_mean = mean(train_error);
    if train_error_mean >= best_train_error
        best_train_error_iter = iter;
        best_train_error = train_error_mean;
        bestNet = Net0;
    end

    % Evaluate fitting quality
    iter = iter+1;
    if mod(iter,params.display_skip)==0
        fprintf(params.printFile,"iter %d: ||h(x)||=%.2e, ||x_dot||=%.2e, ", iter, norm_h_val, norm_sol_dot );
        fprintf(1,"iter %d: ||h(x)||=%.2e, ||x_dot||=%.2e, ", iter, norm_h_val, norm_sol_dot );
        
        fprintf(params.printFile, 'FIT mean (std): %.2f%% (%.2f%%)\n', train_error_mean, std(train_error));
        fprintf(params.printFile, 'FIT: %.2f%%, %.2f%%\n', train_error);
        fprintf(1, 'FIT mean (std): %.2f%% (%.2f%%)\n', train_error_mean, std(train_error));
        fprintf(1, ['FIT: ',repmat('%.2f%%,',1,numExperim),'\n'], train_error);
    end

    if mod(iter,params.ask_continue)==0
        user_decision = input("continue? (y/n): ",'s');
        if user_decision == 'n' || user_decision == 'N'
            break
        end
    end

end
req_time = toc;

fprintf(params.printFile, 'Steps: %d, Time: %.2f s\n', iter, req_time);
fprintf(params.printFile, 'Best network at iteration: %d\n', best_train_error_iter);

end

