%% One-step least-squares estimate for the frictional levitator model.
clear;clc;
folder=fileparts(mfilename('fullpath'));
d=load(fullfile(folder,'data','levitat_data.mat'));

z=d.y(:);u=d.u(:);Ts=d.Ts;rows=numel(z)-2;
acceleration=(z(3:end)-2*z(2:end-1)+z(1:end-2))/Ts^2;
velocity=(z(2:end-1)-z(1:end-2))/Ts;
regressor=[u(1:rows).^2,ones(rows,1),z(1:rows).^2.*velocity];
response=-d.m*z(1:rows).^2.*(acceleration-d.g);
thetaHat=regressor\response;
thetaTrue=[d.Km;d.k0;d.c];

disp(table(thetaTrue,thetaHat,'RowNames',{'Km','k0','c'}));
save(fullfile(folder,'results','ls_result.mat'),'thetaTrue','thetaHat');
