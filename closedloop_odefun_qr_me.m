function [sol_dot, h_val, Net] = closedloop_odefun_qr_me(sol,Net,ut,yt,K,problemInfo)
    rho = K(end);
    numVars = size(sol,1);
    sol_dot = zeros(numVars,1);
    N = problemInfo.N; 
    numParams = Net.numParams;
    x = sol; %x = sol(1:numVars); 
    numExperim = length(ut);

    % Construct Network from optimization variables
    Net = multiLayerNetFromVec(x(1:numParams), Net.n, Net.Nn, Net.dim_u, Net.dim_y);
    % Get output estimates 
    idx_e = numParams;
    for ee = 1:numExperim
        y{ee} = reshape(x(idx_e+1:idx_e+N(ee)*Net.dim_y),Net.dim_y,N(ee))';
        idx_e = idx_e + N(ee)*Net.dim_y;
    end
    
    % Evaluate constraints Jacobian and objective function gradient
    numConstraints = Net.dim_y*(sum(N)-Net.n*numExperim);
    % Allocate memory for grad_f and Jacobian
    Jacob = zeros(numConstraints, numVars);
    grad_f = zeros(numVars,1);
    % Evaluate gradient of f w.r.t. network parameters
    grad_f(1:numParams) = 2*rho*x(1:numParams);
    % Loop over experiments
    row = 0; col = numParams;
    for ee = 1:numExperim
        % Jacobian of h_e(x)
        block = multiLayerJacobianMIMO(Net,y{ee},ut{ee}); 
        % Assign derivatives w.r.t. network parameters
        Jacob(row+1:row+Net.dim_y*(N(ee)-Net.n), 1:numParams) = block(:,1:numParams);
        % Assign derivatives w.r.t. y{ee}
        Jacob(row+1:row+Net.dim_y*(N(ee)-Net.n), col+1:col+Net.dim_y*N(ee)) = block(:,numParams+1:end);
        % Evaluate gradient of f w.r.t. y{ee}
        grad_f(col+1:col+Net.dim_y*N(ee)) = 2*vec((y{ee}-yt{ee})');
        % update row and colum indices
        row = row + Net.dim_y*(N(ee)-Net.n);
        col = col + Net.dim_y*N(ee);
    end
    
    % Evaluation of h(x)
    h_val = zeros(numConstraints,1);
    row = 0; 
    for ee = 1:numExperim
        y_sim = zeros(N(ee)-Net.n,Net.dim_y);
        for tt = Net.n+1:N
            for i = 1:Net.N_layers
                if i == 1
                    t = tanh(Net.W{i}*[vec(y{ee}(tt-1:-1:tt-Net.n,:)'); vec(ut{ee}(tt:-1:tt-Net.n,:)')] + Net.B{i});
                else
                    t = tanh(Net.W{i}*t + Net.B{i});
                end
            end
            y_sim(tt-Net.n,:) = Net.W{end}*t + Net.B{end};
        end
        h_val(row+1:row+Net.dim_y*(N(ee)-Net.n)) = vec( (y_sim - y{ee}(Net.n+1:end,:))' );
        % update row index
        row = row + Net.dim_y*(N(ee)-Net.n);
    end

   Jacob = sparse(Jacob);
   if problemInfo.method == "QR"
       R = qr(Jacob',0);
       R = full(R); J_times_gradf = full(Jacob*grad_f);
       lamb = R\(R'\(-J_times_gradf + K(1)*h_val));
       sol_dot(1:numVars) = -(grad_f + Jacob'*lamb); 
   elseif problemInfo.method == "chol"
       C = chol(full(Jacob*Jacob'));
       J_times_gradf = (Jacob*grad_f);
       lamb = C\(C'\(-J_times_gradf + K(1)*h_val));
       sol_dot(1:numVars) = -(grad_f + Jacob'*lamb); 
   end

end

