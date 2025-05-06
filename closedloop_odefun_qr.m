function [sol_dot, h_val, Net] = closedloop_odefun_qr(sol,Net,ut,yt,K)
    rho = K(end);
    sol_dot = zeros(numel(sol),1);
    N = size(yt,1); 
    numParams = Net.numParams;
    numVars = numParams + Net.dim_y*N;
    x = sol(1:numVars); 
    
    Net = multiLayerNetFromVec(x, Net.n, Net.Nn, Net.dim_u, Net.dim_y);
    y = reshape(x(numParams+1:end),Net.dim_y,N)';

    dh_dx_k = multiLayerJacobianMIMO(Net,y,ut); % -> Jacobian of h(x)
    grad_f = [2*rho*x(1:numParams); 2*vec((y-yt)')]; % -> Gradient of f(x)

    %%%%%%%%%%%%%%%%%%%%%%% Evaluation of h(x) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y_sim = zeros(N-Net.n,Net.dim_y);
    for tt = Net.n+1:N
        for i = 1:Net.N_layers
            if i == 1
                t = tanh(Net.W{i}*[vec(y(tt-1:-1:tt-Net.n,:)'); vec(ut(tt:-1:tt-Net.n,:)')] + Net.B{i});
            else
                t = tanh(Net.W{i}*t + Net.B{i});
            end
        end
        y_sim(tt-Net.n,:) = Net.W{end}*t + Net.B{end};
    end
    h_val = vec( (y_sim - y(Net.n+1:end,:))' );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dh_dx_k = sparse(dh_dx_k);
    R = qr(dh_dx_k',0);
    R = full(R);
    lamb = R\(R'\(-dh_dx_k*grad_f + K(1)*h_val)); % -> lambda
    sol_dot(1:numVars) = -(grad_f + dh_dx_k'*lamb); % -> dx

end

