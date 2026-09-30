function run_cmo_common_tau
% Select one common tau by short warm-ups, then restart all retained CMO fits.
folder=fileparts(mfilename('fullpath')); root=fileparts(folder);
addpath(root,fullfile(root,'nnoe'),fullfile(root,'native'));
out=fullfile(folder,'results','fatigue_common'); if ~exist(out,'dir'),mkdir(out);end
tr=load(fullfile(folder,'data','dataset_training.mat'));
te=load(fullfile(folder,'data','dataset_test.mat'));
u=tr.utrain(1:3500,:); y=tr.ytrain(1:3500,1:2);
ut=te.u_test(1:5000,:); yt=te.y_test(1:5000,1:2);
architectures={10,15,[5 5],[7 7],[5 5 5]}; paperRows=[3 4 6 7 10];
stepSizes=[2e-3;5e-3;1e-2]; regularization=1e-3; gain=1;
scores=-Inf(numel(architectures),numel(stepSizes));
warmRows=cell(numel(architectures)*numel(stepSizes),8); q=0;
for j=1:numel(architectures)
    model=struct('order',3,'hidden',architectures{j});
    for i=1:numel(stepSizes)
        q=q+1;
        try
            r=identify_nnoe(u,y,model,stepSize=stepSizes(i),gain=gain, ...
                regularization=regularization,maxIterations=250,tolerance=1e-4, ...
                displayEvery=0,seed=paperRows(j),fitCheckEvery=10, ...
                fitPatience=10,fitMinDelta=1e-2,minimumIterations=200, ...
                restoreBestFit=true);
            scores(j,i)=r.trainingFit; reason=char(r.stopReason);
        catch ME
            r=struct('trainingFit',-Inf,'elapsedTime',NaN,'iterations',0);
            reason=['failed: ' ME.identifier];
        end
        warmRows(q,:)={paperRows(j),char(strjoin(string(architectures{j}),'x')), ...
            stepSizes(i),r.trainingFit,r.elapsedTime,r.iterations,regularization,reason};
        fprintf('Warm-up row %d tau %.4g: training FIT %.2f%%.\n', ...
            paperRows(j),stepSizes(i),r.trainingFit);
    end
end
% Robust aggregate ranking prevents one architecture from dictating tau.
aggregate=median(scores,1,'omitnan'); [~,best]=max(aggregate);
selectedStep=stepSizes(best);
warmup=cell2table(warmRows,'VariableNames',{'paperRow','hidden','stepSize', ...
    'trainingFit','selectionSeconds','iterations','regularization','stopReason'});
writetable(warmup,fullfile(out,'cmo_common_tau_warmup.csv'));
save(fullfile(out,'cmo_common_tau_warmup.mat'),'warmup','scores','aggregate', ...
    'selectedStep','regularization','gain');
fprintf('Selected one common tau %.4g (aggregate warm-up FIT %.2f%%).\n', ...
    selectedStep,aggregate(best));

rows=cell(numel(architectures),18);
for j=1:numel(architectures)
    model=struct('order',3,'hidden',architectures{j});
    r=identify_nnoe(u,y,model,stepSize=selectedStep,gain=gain, ...
        regularization=regularization,maxIterations=5000,tolerance=1e-4, ...
        displayEvery=0,seed=paperRows(j),fitCheckEvery=10,fitPatience=10, ...
        fitMinDelta=1e-2,minimumIterations=200,restoreBestFit=true);
    if strcmp(r.stopReason,'maximum iterations')
        error('Example3:NoFatigue','Row %d reached cap without fatigue.',paperRows(j));
    end
    pTrain=simulate_nnoe(r.Net,u,y(1:3,:)); [trm,trb]=nnoe_metrics(y,pTrain);
    p=simulate_nnoe(r.Net,ut,yt(1:3,:)); [rm,bfr]=nnoe_metrics(yt,p);
    label=char(strjoin(string(architectures{j}),'x'));
    rows(j,:)={paperRows(j),label,numel(architectures{j}),r.Net.numParams, ...
        trm(1),trm(2),trb(1),trb(2),rm(1),rm(2),bfr(1),bfr(2),mean(bfr), ...
        r.elapsedTime,r.iterations,r.elapsedTime/r.iterations,selectedStep,char(r.stopReason)};
    save(fullfile(out,sprintf('cmo_model_row_%02d.mat',paperRows(j))), ...
        'r','p','pTrain','model','selectedStep','regularization','gain');
    fprintf('CMO row %d %s: %.2f/%.2f%%, %.2fs.\n', ...
        paperRows(j),label,bfr(1),bfr(2),r.elapsedTime);
end
results=cell2table(rows,'VariableNames',{'paperRow','hidden','layers','parameters', ...
    'trainingRMSE1','trainingRMSE2','trainingBFR1','trainingBFR2', ...
    'testRMSE1','testRMSE2','testBFR1','testBFR2','meanTestBFR', ...
    'trainingSeconds','iterations','secondsPerIteration','stepSize','stopReason'});
writetable(results,fullfile(out,'cmo_common_tau_models.csv'));
end
