function [Net, numParams] = initialize_nnoe(n, Nn, dim_u, dim_y)
    
    % Preallocation
    N_layers = length(Nn);
    W = cell(N_layers+1, 1);
    B = cell(N_layers+1, 1);
    numParams = 0;
    
    % Inizialization
    for i = 1:N_layers+1
        if i == 1
            W{i} = randn(Nn(i), n*dim_y + (n+1)*dim_u);
            B{i} = randn(Nn(i), 1);
        elseif i == N_layers+1
            W{i} = randn(dim_y, Nn(i-1));
            B{i} = randn(dim_y,1);
        else
            W{i} = randn(Nn(i), Nn(i-1));
            B{i} = randn(Nn(i), 1);
        end
        numParams = numParams + numel(W{i}) + numel(B{i});
    end
    
    W1y = W{1}(:,1:n*dim_y);
    W1u = W{1}(:,(n*dim_y)+1:end);
    
    % Net structure
    Net.n = n;
    Net.Nn = Nn;
    Net.N_layers = N_layers;
    Net.dim_u = dim_u;
    Net.dim_y = dim_y;
    Net.W = W;
    Net.W1y = W1y;
    Net.W1u = W1u;
    Net.B = B;
    Net.numParams = numParams;
end

