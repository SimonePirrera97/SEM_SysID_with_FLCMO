function runs = method_flcmo(cfg)
%METHOD_FLCMO Constrained-manifold NNOE with measured y as the y initial point.
addpath(cfg.root,fullfile(cfg.root,'nnoe'),fullfile(cfg.root,'native'));
model=struct('order',cfg.order,'hidden',cfg.hidden);
rates=[1e-4; 5e-4; 1e-3];
gain=1; regularization=1e-3;
cv=zeros(numel(rates),6);
for i=1:numel(rates)
    % Same network initial point used by all validation candidates.
    x0=[cfg.theta0(:,1);cfg.flcmoOutput0(:,1)];
    r=timed_identify(cfg,model,x0,rates(i),gain,regularization,250);
    p=simulate_nnoe(r.Net,cfg.validation.u,cfg.validation.y(1:cfg.order,:));
    [rmse,fit]=es1_metrics(cfg.validation.y,p,cfg.order);
    cv(i,:)=[i rates(i) rmse fit r.elapsedTime r.timedOut];
    es1_log(cfg,sprintf('FLCMO hyperparameter candidate %d',i),sprintf('stepSize=%.6g, validation RMSE=%.6g, FIT=%.6g%%',rates(i),rmse,fit));
end
cv=array2table(cv,'VariableNames',{'candidate','stepSize','validationRMSE','validationFIT','seconds','timedOut'});
[~,ibest]=max(cv.validationFIT); selectedRate=rates(ibest);
es1_log(cfg,'FLCMO hyperparameter selection',sprintf('candidate=%d, stepSize=%.6g, validation FIT=%.6g%%',ibest,selectedRate,cv.validationFIT(ibest)));

data=zeros(cfg.runs,11);
for run=1:cfg.runs
    % FLCMO alone optimizes y as well; use the successful default random
    % scale for those auxiliary variables while preserving common weights.
    x0=[cfg.theta0(:,run);cfg.flcmoOutput0(:,run)];
    r=timed_identify(cfg,model,x0,selectedRate,gain,regularization,2000);
    p=simulate_nnoe(r.Net,cfg.test.u,cfg.test.y(1:cfg.order,:));
    [rmse,fit]=es1_metrics(cfg.test.y,p,cfg.order);
    data(run,:)=[run rmse fit r.trainingFit r.elapsedTime r.iterations selectedRate gain regularization r.timedOut cfg.fitTimeoutSeconds];
    es1_log(cfg,sprintf('FLCMO identification from initialization %d',run),sprintf('test RMSE=%.6g, FIT=%.6g%%, iterations=%d, time=%.6g s',rmse,fit,r.iterations,r.elapsedTime));
end
runs=array2table(data,'VariableNames',{'run','testRMSE','testFIT','trainingFIT','seconds','iterations','stepSize','gain','regularization','timedOut','timeoutSeconds'});
es1_log(cfg,'Completed FLCMO run results',sprintf('%d rows retained for aggregate CSV',height(runs)));
end

function r=timed_identify(cfg,model,x0,rate,gain,regularization,maxIterations)
start=tic; completed=0; timedOut=false; chunkSize=10;
while completed<maxIterations
    chunk=min(chunkSize,maxIterations-completed);
    r=identify_nnoe(cfg.train.u,cfg.train.y,model,x0=x0, ...
        stepSize=rate,gain=gain,regularization=regularization, ...
        maxIterations=chunk,displayEvery=0,minimumIterations=chunk);
    completed=completed+r.iterations; x0=r.x;
    if toc(start)>=cfg.fitTimeoutSeconds, timedOut=true; break; end
    if r.iterations<chunk || r.stopReason=="numerical tolerance", break; end
end
r.iterations=completed; r.elapsedTime=toc(start); r.timedOut=timedOut;
if timedOut, r.stopReason="wall-time limit"; end
end
