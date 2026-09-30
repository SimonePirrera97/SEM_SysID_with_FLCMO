function h = nnoe_constraint_values(Net,y,u)
%NNOE_CONSTRAINT_VALUES Evaluate M_theta(history)-y_k.
N = size(y,1);
hmat = zeros(N-Net.n,Net.dim_y);
for k = Net.n+1:N
    for layer = 1:Net.N_layers
        if layer == 1
            a = tanh(Net.W{layer} * ...
                [reshape(y(k-1:-1:k-Net.n,:).',[],1); ...
                 reshape(u(k:-1:k-Net.n,:).',[],1)] + Net.B{layer});
        else
            a = tanh(Net.W{layer}*a + Net.B{layer});
        end
    end
    hmat(k-Net.n,:) = Net.W{end}*a + Net.B{end} - y(k,:).';
end
h = reshape(hmat.',[],1);
end
