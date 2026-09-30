function run_revised_nnoe
% Revised Example 3 FL-CMO runs: all submitted-paper architectures.
folder=fileparts(mfilename('fullpath')); root=fileparts(folder);
addpath(root,fullfile(root,'nnoe'),fullfile(root,'native'));
out=fullfile(folder,'results','revised'); if ~exist(out,'dir'),mkdir(out);end
tr=load(fullfile(folder,'data','dataset_training.mat')); te=load(fullfile(folder,'data','dataset_test.mat'));
u=tr.utrain(1:3500,:); y=tr.ytrain(1:3500,1:2); split=2800;
tuneEnd=1400; tuneValidationEnd=2100;
% K is fixed to one. The global step was selected previously. Architectures
% whose held-out fits failed are retuned separately on the chronological
% validation split. Every final fit is restarted, excluding all tuning time.
defaultStep=1e-2;
stepSizes=[1e-3;2e-3;5e-3;1e-2;2e-2];
gain=1; regularization=1e-4;
architectures={5,8,10,15,[3 3],[5 5],[7 7],[10 10],[3 3 3],[5 5 5],[7 7 7]};
failedRows=[8 10 11]; selectedSteps=defaultStep*ones(numel(architectures),1);
cvRows=cell(numel(failedRows)*numel(stepSizes),15); cvRow=0;
for j=1:numel(failedRows)
    paperRow=failedRows(j); hidden=architectures{paperRow};
    candidateScores=-Inf(numel(stepSizes),1);
    for i=1:numel(stepSizes)
        cvRow=cvRow+1;
        try
            pilot=struct('order',3,'hidden',hidden);
            r=identify_nnoe(u(1:tuneEnd,:),y(1:tuneEnd,:),pilot, ...
                stepSize=stepSizes(i),gain=gain,regularization=regularization, ...
                maxIterations=800,tolerance=1e-4,displayEvery=0,seed=400+paperRow, ...
                fitCheckEvery=10,fitPatience=8,fitMinDelta=1e-2, ...
                minimumIterations=100,restoreBestFit=true);
            p=simulate_nnoe(r.Net,u(tuneEnd+1:tuneValidationEnd,:), ...
                y(tuneEnd+(1:3),:));
            [rmse,bfr]=nnoe_metrics(y(tuneEnd+1:tuneValidationEnd,:),p);
            if ~all(isfinite([rmse bfr]))
                error('Example3:NonfiniteCandidate','Non-finite validation metric.');
            end
            reason=char(r.stopReason); candidateScores(i)=mean(bfr);
            fprintf('FL-CMO row %d tau %.3g: BFR %.2f/%.2f%%, stop %d.\n', ...
                paperRow,stepSizes(i),bfr(1),bfr(2),r.iterations);
        catch ME
            warning('Example3:StepCandidateFailed','row %d tau %.3g failed: %s', ...
                paperRow,stepSizes(i),ME.message);
            rmse=[Inf Inf]; bfr=[-Inf -Inf];
            r=struct('elapsedTime',NaN,'iterations',0); reason=['failed: ' ME.identifier];
        end
        cvRows(cvRow,:)={paperRow,char(strjoin(string(hidden),'x')),i,stepSizes(i), ...
            gain,regularization,rmse(1),rmse(2),bfr(1),bfr(2),mean(bfr), ...
            r.elapsedTime,r.iterations,r.elapsedTime/max(1,r.iterations),reason};
    end
    [~,best]=max(candidateScores); selectedSteps(paperRow)=stepSizes(best);
end
cv=cell2table(cvRows,'VariableNames',{'paperRow','hidden','candidate','stepSize', ...
    'gain','regularization','validationRMSE1','validationRMSE2', ...
    'validationBFR1','validationBFR2','meanValidationBFR','trainingSeconds', ...
    'iterations','secondsPerIteration','stopReason'});
writetable(cv,fullfile(out,'nnoe_algorithm_cv.csv'));
save(fullfile(out,'nnoe_algorithm_cv.mat'),'cv','selectedSteps','gain','regularization');
ut=te.u_test(1:5000,:); yt=te.y_test(1:5000,1:2); rows=cell(numel(architectures),18);
for i=1:numel(architectures)
    model=struct('order',3,'hidden',architectures{i});
    selectedStep=selectedSteps(i);
    r=identify_nnoe(u,y,model,stepSize=selectedStep,gain=gain, ...
        regularization=regularization,maxIterations=2000,tolerance=1e-4, ...
        displayEvery=0,seed=i,fitCheckEvery=10,fitPatience=15, ...
        fitMinDelta=1e-2,minimumIterations=200,restoreBestFit=true);
    pTrain=simulate_nnoe(r.Net,u,y(1:3,:)); [trainRMSE,trainBFR]=nnoe_metrics(y,pTrain);
    p=simulate_nnoe(r.Net,ut,yt(1:3,:)); [rmse,bfr]=nnoe_metrics(yt,p);
    label=strjoin(string(architectures{i}),'x'); parameters=r.Net.numParams;
    rows(i,:)={i,char(label),numel(architectures{i}),parameters, ...
        trainRMSE(1),trainRMSE(2),trainBFR(1),trainBFR(2), ...
        rmse(1),rmse(2),bfr(1),bfr(2),mean(bfr),r.elapsedTime, ...
        r.iterations,r.elapsedTime/r.iterations,selectedStep,char(r.stopReason)};
    save(fullfile(out,sprintf('nnoe_model_%02d.mat',i)),'r','p','pTrain', ...
        'trainRMSE','trainBFR','rmse','bfr','model','selectedStep','gain','regularization');
    fprintf('FL-CMO %s: best iter %d, stop %d, test BFR %.2f/%.2f%% (%s).\n', ...
        label,r.bestMonitorIteration,r.iterations,bfr(1),bfr(2),r.stopReason);
end
results=cell2table(rows,'VariableNames',{'paperRow','hidden','layers','parameters', ...
    'trainingRMSE1','trainingRMSE2','trainingBFR1','trainingBFR2', ...
    'testRMSE1','testRMSE2','testBFR1','testBFR2','meanTestBFR', ...
    'trainingSeconds','iterations','secondsPerIteration','stepSize','stopReason'});
writetable(results,fullfile(out,'nnoe_all_paper_models.csv'));
fprintf('Example 3 NNOE: evaluated all %d paper architectures.\n',height(results));
end
