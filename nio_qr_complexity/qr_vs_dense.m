clear; close all; clc;

N = 5000;
p = 2; phi = 2;
n_th = 50;

numCns = p*(N-phi);
dimVar = n_th + p*N;

r = randn(numCns,1);

J = zeros(numCns,dimVar);
J(:,1:n_th) = randn(numCns,n_th);
idxc = n_th+1;
for ii = 1:N-phi
    J((ii-1)*p+1:ii*p, idxc:idxc+(phi+1)*p-1) = randn(p,(phi+1)*p);
    idxc = idxc + p;
end
spy(J)

tic
s_dense = (J*J')\r;
time_dense = toc;

J = sparse(J);
tic 
R = qr(J',0);
s_sparse = R\(R'\r);
time_sparse = toc;

% max(abs(s_dense-s_sparse))  % =0, ok
format short e;
disp(table(N, time_dense, time_sparse))

    %     N         time_dense    time_sparse
    % __________    __________    ___________
    %     
    % 1.0000e+03    5.0853e-02    3.0902e-02 
    % 5.0000e+03    3.6866e+00    9.9071e-01 
    % 1.0000e+04    2.2683e+01    4.0379e+00 