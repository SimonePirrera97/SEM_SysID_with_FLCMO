function y = simulateMultiLayerNNOE(Net, u, y0)
    N = size(u,1); 
    n = Net.n; dim_y = Net.dim_y;
    y = zeros(N,dim_y);
    
    % Initial conditions
    if nargin == 3 && size(y0,1)==n && size(y0,2)==dim_y
        y(1:n,:) = y0;
    else 
        y(1:n,:) = zeros(n,dim_y);
    end

    % Simulation with for loop
    for k = n+1:N
        for i = 1:Net.N_layers
            if i == 1
                t = tanh(Net.W{i}*[vec(y(k-1:-1:k-n, :)'); vec(u(k:-1:k-n,:)')] + Net.B{i});
            else
                t = tanh(Net.W{i}*t + Net.B{i});
            end
        end
        y(k,:) = (Net.W{end}*t + Net.B{end})';
    end

end
