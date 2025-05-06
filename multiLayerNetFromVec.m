function Net = multiLayerNetFromVec(theta, n, Nn, dim_u, dim_y)

    % Preallocation
    W = cell(length(Nn)+1, 1);
    B = cell(length(Nn)+1, 1);
    
    idx = 1;
    
    for i = 1:length(Nn)+1
        if i == 1
            W1y = reshape(theta(idx:idx+Nn(i)*n*dim_y-1), n*dim_y, Nn(i))';
            idx = idx + Nn(i)*n*dim_y;
            W1u = reshape(theta(idx:idx+Nn(i)*(n+1)*dim_u-1), (n+1)*dim_u, Nn(i))';
            idx = idx + Nn(i)*(n+1)*dim_u;
            W{i} = [W1y, W1u];
            B{i} = theta(idx:idx+Nn(i)-1);
            idx = idx + Nn(i);
            
        elseif i == length(Nn)+1
            We=theta(idx:idx+dim_y*Nn(i-1)-1);
            W{end} = reshape(We,numel(We)/dim_y,dim_y)';
            idx = idx + Nn(i-1)*dim_y;
            B{end} = theta(idx:idx+dim_y-1);
            
        else
            W{i} = reshape(theta(idx:idx+Nn(i-1)*Nn(i)-1), Nn(i-1), Nn(i))';
            idx = idx + Nn(i-1)*Nn(i);
            B{i} = theta(idx:idx+Nn(i)-1);
            idx = idx + Nn(i);
        end
    end

    % Net structure
    Net.n = n;
    Net.Nn = Nn;
    Net.N_layers = length(Nn);
    Net.W = W;
    Net.W1y = W1y;
    Net.W1u = W1u;
    Net.B = B;
    Net.dim_u = dim_u;
    Net.dim_y = dim_y;
    Net.numParams = length(theta);
end