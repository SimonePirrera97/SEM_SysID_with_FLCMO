function runs = method_nnsysid_nnoe(cfg)
%METHOD_NNSYSID_NNOE NNOE comparison using NNSYSID20's nnoe().
assert(exist(fullfile(cfg.nnsysid,'nnoe.m'),'file')==2,'NNSYSID20 nnoe.m was not found.');
addpath(cfg.nnsysid);
NetDef=[repmat('H',1,cfg.hidden); 'L' repmat('-',1,cfg.hidden-1)];
NN=[cfg.order cfg.order+1 0]; % y(k-1:k-4), u(k:k-4)
lambdas=[0.1; 1; 10];
weightDecay=1e-3; % Same regularization coefficient used by every method.
cv=zeros(numel(lambdas),6);
[W1,W2]=weights(cfg.theta0(:,1),cfg);
for i=1:numel(lambdas)
    [A,B,~,~,~,elapsed,timedOut]=timed_nnoe(cfg,NetDef,NN,W1,W2,lambdas(i),weightDecay,100);
    p=simulate(A,B,cfg.validation.u,cfg.validation.y,cfg);
    [rmse,fit]=es1_metrics(cfg.validation.y,p,cfg.order);
    cv(i,:)=[i lambdas(i) rmse fit elapsed timedOut];
    es1_log(cfg,sprintf('NNSYSID NNOE hyperparameter candidate %d',i),sprintf('lambda=%.6g, validation RMSE=%.6g, FIT=%.6g%%, time=%.6g s',lambdas(i),rmse,fit,elapsed));
end
cv=array2table(cv,'VariableNames',{'candidate','lambda','validationRMSE','validationFIT','seconds','timedOut'});
[~,ibest]=max(cv.validationFIT); selectedLambda=lambdas(ibest);
es1_log(cfg,'NNSYSID NNOE hyperparameter selection',sprintf('candidate=%d, lambda=%.6g, validation FIT=%.6g%%',ibest,selectedLambda,cv.validationFIT(ibest)));

data=zeros(cfg.runs,9);
for run=1:cfg.runs
    [W1,W2]=weights(cfg.theta0(:,run),cfg);
    [W1,W2,criterion,iterations,lambda,elapsed,timedOut]=timed_nnoe(cfg,NetDef,NN,W1,W2,selectedLambda,weightDecay,500);
    p=simulate(W1,W2,cfg.test.u,cfg.test.y,cfg);
    [rmse,fit]=es1_metrics(cfg.test.y,p,cfg.order);
    data(run,:)=[run rmse fit elapsed iterations lambda criterion(end) timedOut cfg.fitTimeoutSeconds];
    es1_log(cfg,sprintf('NNSYSID NNOE identification from initialization %d',run),sprintf('test RMSE=%.6g, FIT=%.6g%%, iterations=%d, time=%.6g s',rmse,fit,iterations,elapsed));
end
runs=array2table(data,'VariableNames',{'run','testRMSE','testFIT','seconds','iterations','finalLambda','finalCriterion','timedOut','timeoutSeconds'});
es1_log(cfg,'Completed NNSYSID NNOE run results',sprintf('%d rows retained for aggregate CSV',height(runs)));
end

function [W1,W2,criterion,iterations,lambda,elapsed,timedOut]=timed_nnoe(cfg,NetDef,NN,W1,W2,lambda,weightDecay,maxIterations)
start=tic; iterations=0; criterion=[]; timedOut=false; chunkSize=10;
while iterations<maxIterations
    chunk=min(chunkSize,maxIterations-iterations);
    tr=settrain; tr=settrain(tr,'maxiter',chunk,'D',weightDecay, ...
        'lambda',lambda,'skip',cfg.order,'infolevel',0);
    [W1,W2,currentCriterion,currentIterations,lambda]=nnoe( ...
        NetDef,NN,W1,W2,tr,cfg.train.y',cfg.train.u');
    criterion=[criterion currentCriterion(:)']; %#ok<AGROW>
    iterations=iterations+currentIterations;
    if toc(start)>=cfg.fitTimeoutSeconds, timedOut=true; break; end
    if currentIterations<chunk, break; end
end
elapsed=toc(start);
end

function [W1,W2]=weights(theta,cfg)
i=1; ny=cfg.hidden*cfg.order; nu=cfg.hidden*(cfg.order+1);
Wy=reshape(theta(i:i+ny-1),cfg.order,cfg.hidden)'; i=i+ny;
Wu=reshape(theta(i:i+nu-1),cfg.order+1,cfg.hidden)'; i=i+nu;
b=theta(i:i+cfg.hidden-1); i=i+cfg.hidden;
v=theta(i:i+cfg.hidden-1)'; i=i+cfg.hidden;
W1=[Wy Wu b]; W2=[v theta(i)];
end

function p=simulate(W1,W2,u,y,cfg)
p=y; n=cfg.order;
for k=n+1:numel(y)
    phi=[p(k-1:-1:k-n)' u(k:-1:k-n)'];
    p(k)=W2*[tanh(W1*[phi';1]);1];
end
end
