%% Short open-loop validation for all Example 4 parameter estimates.
clear;clc;
folder=fileparts(mfilename('fullpath'));
d=load(fullfile(folder,'data','levitat_data.mat'));
validationInput=d.u_validation(:);
validationTarget=d.y_validation_true(:);
models={}; names={};

flcmo=load(fullfile(folder,'results','optimized_result_fatigue.mat'));
models{end+1}=flcmo.estimatedParameters; names{end+1}='FL-CMO fatigue';
ls=load(fullfile(folder,'results','ls_result.mat'));
models{end+1}=ls.thetaHat; names{end+1}='LS';

% Estimates from the reproducible 10^4-epoch Adam runs.
models{end+1}=[2.2025715189502476e-4;2.0170753658549876e-5;3.412982457089824e-2];
names{end+1}='Adam full BPTT';
models{end+1}=[2.0946538822914823e-4;2.0421882487490245e-5;3.698964089978169e-2];
names{end+1}='Adam length-128 batches';

fit=zeros(numel(models),1);rmse=fit;
for model=1:numel(models)
    prediction=simulateOpenLoop(models{model},validationInput, ...
        validationTarget(1:2),d.Ts,d.m,d.g);
    error=prediction-validationTarget;
    fit(model)=1-sum(error.^2)/sum((validationTarget-mean(validationTarget)).^2);
    rmse(model)=sqrt(mean(error.^2));
end
report=table(names(:),fit,rmse,'VariableNames',{'Model','FIT','RMSE'});
disp(report);
save(fullfile(folder,'results','validation_accuracy.mat'),'report');

function prediction=simulateOpenLoop(parameters,input,initialOutput,Ts,mass,gravity)
prediction=zeros(size(input)); prediction(1:2)=initialOutput;
for index=1:numel(input)-2
    gap=prediction(index);
    velocity=(prediction(index+1)-gap)/Ts;
    acceleration=gravity-(parameters(1)*input(index)^2+parameters(2)) ...
        /(mass*gap^2)-parameters(3)*velocity/mass;
    prediction(index+2)=2*prediction(index+1)-gap+Ts^2*acceleration;
end
end
