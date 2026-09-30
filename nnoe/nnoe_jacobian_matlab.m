function Jmc = nnoe_jacobian_matlab(Net, y, ut)
    % MULTI-LAYER JACOBIAN
    % Evaluates the jacobian of the constraints vector for the constrained
    % optimization problem arising when training of a MIMO multi-layer NNOE
    
    % Extract data from Net structure
    W = Net.W;
    B = Net.B;
    N = size(y,1);
    dim_u = Net.dim_u;
    dim_y = Net.dim_y;
    N_layers = Net.N_layers;
    Nn = Net.Nn;
    n = Net.n;

    % Pre-allocate memory for jacobian calculations
    grad_h_W = cell(N_layers+1,1);
    grad_h_B = cell(N_layers+1,1);
    numEquations = (N-n)*dim_y;

    for i = 1:N_layers+1
        if i == 1
            grad_h_W{i} = zeros(Nn(i)*(n*dim_y+(n+1)*dim_u), numEquations);
            grad_h_B{i} = zeros(Nn(i), numEquations);
        elseif i == N_layers+1
            grad_h_W{i} = zeros(dim_y*Nn(i-1), numEquations);
            grad_h_B{i} = zeros(dim_y, numEquations);
        else
            grad_h_W{i} = zeros(Nn(i-1)*Nn(i), numEquations);
            grad_h_B{i} = zeros(Nn(i), numEquations);
        end
    end
    W1y = W{1}(:,1:dim_y*n);

    grad_h_Wy = zeros(Nn(1)*n*dim_y, numEquations);
    grad_h_Wu = zeros(Nn(1)*(n+1)*dim_u, numEquations);
    
    grad_h_y = zeros(N*dim_y,numEquations);

    % Evaluate jacobian
    t = cell(N_layers,1);
    dt = cell(N_layers,1);

    grad_h_y_row = 0;

    for k = n+1:N
        
        idx = k-n;
        for i = 1:N_layers
            if i == 1
                outputHistory=reshape(y(k-1:-1:k-n,:).',[],1);
                inputHistory=reshape(ut(k:-1:k-n,:).',[],1);
                t{i}=tanh(W{i}*[outputHistory;inputHistory]+B{i});
            else
                t{i} = tanh(W{i}*t{i-1} + B{i});
            end
            dt{i} = 1 - t{i}.^2;
        end
        % DEBUG: a = 5;
        grad_h_W{end}(1:dim_y*Nn(end),(idx-1)*dim_y+1:idx*dim_y) = kron(eye(dim_y),t{end});
        
        grad_h_B{end}(:,(idx-1)*dim_y+1:idx*dim_y) = eye(dim_y);
        
    % DEBUG: end
        for idx_y = 1:dim_y
            
            temp = W{end}(idx_y,:)'.*dt{end};

            if N_layers > 1
            
            gW = temp*t{end-1}';
            grad_h_W{end-1}(:,(idx-1)*dim_y+idx_y) = reshape(gW', numel(gW), 1);
            grad_h_B{end-1}(:,(idx-1)*dim_y+idx_y) = temp;
        
            for i = N_layers:-1:2
                temp = (W{i}'*temp).*dt{i-1};
                if i > 2
                    gW = temp*t{i-2}';
                    grad_h_W{i-1}(:,(idx-1)*dim_y+idx_y) = reshape(gW', numel(gW), 1);
                    grad_h_B{i-1}(:,(idx-1)*dim_y+idx_y) = temp;
                end
            end
            end

            gWy=temp*reshape(y(k-1:-1:k-n,:).',[],1).';
            grad_h_Wy(:,(idx-1)*dim_y+idx_y) = reshape(gWy', numel(gWy), 1);
            gWu=temp*reshape(ut(k:-1:k-n,:).',[],1).';
            grad_h_Wu(:,(idx-1)*dim_y+idx_y) = reshape(gWu', numel(gWu), 1);
            grad_h_B{1}(:,(idx-1)*dim_y+idx_y) = temp;

            %if dim_y == 1
            %    grad_h_y((k-n-1)*dim_y+dim_y*n:-1:(k-n-1)*dim_y+1,grad_h_y_row+idx_y) = W1y'*temp;
            %else
                grad_h_y((k-n-1)*dim_y+dim_y*n:-1:(k-n-1)*dim_y+1,grad_h_y_row+idx_y) = ...
                    reshape(flip(reshape(W1y'*temp,dim_y, ...
                    numel(W1y'*temp)/dim_y).',2).',[],1);
            %end
        end
        grad_h_y((k-1)*dim_y+1:k*dim_y,grad_h_y_row+1:grad_h_y_row+dim_y) = -eye(dim_y);
        grad_h_y_row = grad_h_y_row+dim_y;
    end

    % Collect results in a unique matrix
    Jmc = [grad_h_Wy', grad_h_Wu', grad_h_B{1}'];
    for i = 2:N_layers+1
        Jmc = cat(2, Jmc, [grad_h_W{i}', grad_h_B{i}']);
    end
    Jmc = cat(2, Jmc, grad_h_y');
    %size(grad_h_y')
end
