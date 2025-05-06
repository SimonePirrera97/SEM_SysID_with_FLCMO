%% Identification of levitator gray-box parameters by FL-CMO.
% Simone Pirrera
% Oct 16, 2024

clear; close all; clc;
load('levitat_data.mat');
km_true = Km;
k0_true = k0;

N = 200; numParams = 2;
yt = y(1:N); u = u(1:N);

max_iter = 10000;
x = randn(numParams+N,max_iter);

Kp = 2; dt = 1e-3;

for iter = 1:max_iter-1
    km = x(1,iter); k0 = x(2,iter); 
    y = x(3:end,iter);
    df = [zeros(numParams,1);2*(y-yt)];
    J = jacob(x(:,iter),u,Ts,m,g);
    % Evaluate constraints 
    rr = (m*y(1:N-2).^2).*( (y(3:N)-2*y(2:N-1)+y(1:N-2))/Ts^2 -g);
    i2 = u(1:N-2).^2;
    h = rr + km*i2 + k0;

    xdot = -df-J'*((J*J')\(J*df+Kp*h));
    x(:,iter+1) = x(:,iter) + dt*xdot;
end

figure, plot(x(1:2,:)')

theta_true = [km_true,k0_true]';
theta_hat = x(1:numParams,end);

fprintf(1, "True parameters = [%.3e, %.3e]\n", [km_true,k0_true]');
fprintf(1, "Estimated parameters = [%.3e, %.3e]\n", x(1:numParams,end));
fprintf(1, "Norm of the constraints = %.3f\n", norm(h));

disp(table(theta_true,theta_hat))


function J = jacob(x,u,Ts,m,g)
    km = x(1); k0 = x(2); 
    y = x(3:end);
    N = length(y); n = 2;   
    J = zeros(N-n,length(x));
    J(:,1) = u(1:N-2).^2;
    J(:,2) = ones(N-n,1);
    %J(:,3) = -Ts^2*(km*u(1:N-2).^2+k0)./(m*y(1:N-2).^2);
    numParams = 2;
    J(:,numParams+1:numParams+N-2) = diag( (m/Ts^2)*y(1:N-2).^2 );
    J(:,numParams+2:numParams+N-1) = J(:,numParams+2:numParams+N-1) + ...
        + diag( -2*(m/Ts^2)*y(1:N-2).^2 );
    J(:,numParams+3:numParams+N) = J(:,numParams+3:numParams+N) + ...
        + diag( (m/Ts^2).*y(1:N-2).*(2*y(3:N)-4*y(2:N-1)+3*y(1:N-2))-2*m*g*y(1:N-2) );
end


