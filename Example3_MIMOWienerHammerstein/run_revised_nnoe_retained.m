function run_revised_nnoe_retained
% Retained Example-3 FL-CMO models with per-architecture step selection.
folder=fileparts(mfilename('fullpath')); root=fileparts(folder);
addpath(root,fullfile(root,'nnoe'),fullfile(root,'native'));
out=fullfile(folder,'results','full_sequence'); if ~exist(out,'dir'),mkdir(out);end
tr=load(fullfile(folder,'data','dataset_training.mat'));
te=load(fullfile(folder,'data','dataset_test.mat'));
u=tr.utrain(1:3500,:); y=tr.ytrain(1:3500,1:2);
ut=te.u_test(1:5000,:); yt=te.y_test(1:5000,1:2);
architectures={10,15,[5 5],[7 7],[5 5 5]};
paperRows=[3 4 6 7 10];
stepSizes=[5e-4;1e-3;2e-3;5e-3;1e-2;2e-2];
gain=1; regularization=1e-3; tuneEnd=1400; validationEnd=2100;
cvRows=cell(numel(architectures)*numel(stepSizes),15); cvRow=0;
selectedSteps=zeros(numel(architectures),1);
for j=1:numel(architectures)
    hidden=architectures{j}; scores=-Inf(numel(stepSizes),1);
    for i=1:numel(stepSizes)
        cvRow=cvRow+1; model=struct('order',3,'hidden',hidden);
        try
            r=identify_nnoe(u(1:tuneEnd,:),y(1:tuneEnd,:),model, ...
                stepSize=stepSizes(i),gain=gain,regularization=regularization, ...
                maxIterations=1000,tolerance=1e-4,displayEvery=0,seed=paperRows(j), ...
                fitCheckEvery=10,fitPatience=12,fitMinDelta=1e-2, ...
                minimumIterations=200,restoreBestFit=true);
            p=simulate_nnoe(r.Net,u(tuneEnd+1:validationEnd,:), ...
                y(tuneEnd+(1:3),:));
            [rmse,bfr]=nnoe_metrics(y(tuneEnd+1:validationEnd,:),p);
            if ~all(isfinite([rmse bfr])), error('nonfinite metrics'); end
            scores(i)=mean(bfr); reason=char(r.stopReason);
            fprintf('CMO row %d tau %.4g: %.2f/%.2f%% (%d iterations).\n', ...
                paperRows(j),stepSizes(i),bfr(1),bfr(2),r.iterations);
        catch ME
            rmse=[Inf Inf]; bfr=[-Inf -Inf];
            r=struct('elapsedTime',NaN,'iterations',0); reason=['failed: ' ME.identifier];
        end
        cvRows(cvRow,:)={paperRows(j),char(strjoin(string(hidden),'x')),i, ...
            stepSizes(i),gain,regularization,rmse(1),rmse(2),bfr(1),bfr(2), ...
            mean(bfr),r.elapsedTime,r.iterations, ...
            r.elapsedTime/max(1,r.iterations),reason};
    end
    [~,best]=max(scores); selectedSteps(j)=stepSizes(best);
end
cv=cell2table(cvRows,'VariableNames',{'paperRow','hidden','candidate','stepSize', ...
    'gain','regularization','validationRMSE1','validationRMSE2', ...
    'validationBFR1','validationBFR2','meanValidationBFR','selectionSeconds', ...
    'iterations','secondsPerIteration','stopReason'});
writetable(cv,fullfile(out,'cmo_step_cv.csv'));
save(fullfile(out,'cmo_step_cv.mat'),'cv','selectedSteps','regularization','gain');

rows=cell(numel(architectures),18);
for j=1:numel(architectures)
    hidden=architectures{j}; model=struct('order',3,'hidden',hidden);
    selectedStep=selectedSteps(j);
    r=identify_nnoe(u,y,model,stepSize=selectedStep,gain=gain, ...
        regularization=regularization,maxIterations=2500,tolerance=1e-4, ...
        displayEvery=0,seed=paperRows(j),fitCheckEvery=10,fitPatience=20, ...
        fitMinDelta=1e-2,minimumIterations=300,restoreBestFit=true);
    pTrain=simulate_nnoe(r.Net,u,y(1:3,:));
    [trainRMSE,trainBFR]=nnoe_metrics(y,pTrain);
    p=simulate_nnoe(r.Net,ut,yt(1:3,:)); [rmse,bfr]=nnoe_metrics(yt,p);
    label=char(strjoin(string(hidden),'x'));
    rows(j,:)={paperRows(j),label,numel(hidden),r.Net.numParams, ...
        trainRMSE(1),trainRMSE(2),trainBFR(1),trainBFR(2), ...
        rmse(1),rmse(2),bfr(1),bfr(2),mean(bfr),r.elapsedTime,r.iterations, ...
        r.elapsedTime/r.iterations,selectedStep,char(r.stopReason)};
    save(fullfile(out,sprintf('cmo_model_row_%02d.mat',paperRows(j))), ...
        'r','p','pTrain','model','selectedStep','regularization','gain');
    fprintf('CMO final row %d %s tau %.4g: %.2f/%.2f%%, %.2fs.\n', ...
        paperRows(j),label,selectedStep,bfr(1),bfr(2),r.elapsedTime);
end
results=cell2table(rows,'VariableNames',{'paperRow','hidden','layers','parameters', ...
    'trainingRMSE1','trainingRMSE2','trainingBFR1','trainingBFR2', ...
    'testRMSE1','testRMSE2','testBFR1','testBFR2','meanTestBFR', ...
    'trainingSeconds','iterations','secondsPerIteration','stepSize','stopReason'});
writetable(results,fullfile(out,'cmo_retained_models.csv'));
end
