function summary = main_smart_search(options)
%MAIN_SMART_SEARCH Ordered, restart-safe Bouc--Wen NNOE campaign.
%
% Default campaign (36 primary fits, one initialization each):
%   9 ordered (order, architecture) pairs x K=[10,1] x
%   lambda=[1e-3,5e-3]. For each primary fit, four short runs from the
%   same initial point rank tau candidates and reject divergent choices.
%   The best stable tau is then restarted and trained until FIT fatigue;
%   the next tau is tried if the full run diverges. Every accepted model is
%   evaluated on both validation and benchmark-test records. Selection is
%   based strictly on validation BFR, never on test BFR.
%
% From the repository root:
%   matlab -batch "addpath('code_new/Example2_BoucWen'); main_smart_search"
%
% Short plumbing check (separate output directory):
%   matlab -batch "addpath('code_new/Example2_BoucWen'); ...
%       main_smart_search(quickTest=true,resume=false)"

arguments
    options.trainingSamples (1,1) double {mustBeInteger,mustBePositive} = 12000
    options.validationSamples (1,1) double {mustBeInteger,mustBePositive} = 5000
    options.orders (1,:) double {mustBeInteger,mustBePositive} = [3 4 5]
    options.architectures (1,:) cell = {[8 8],[6 6],[10 10]}
    options.gains (1,:) double {mustBePositive} = [10 1]
    options.regularizations (1,:) double {mustBePositive} = [1e-3 5e-3]
    options.stepSizes (1,:) double {mustBePositive} = [2e-3 1e-3 5e-4 5e-3]
    options.tauProbeIterations (1,1) double {mustBeInteger,mustBePositive} = 75
    options.numInitializations (1,1) double {mustBeInteger,mustBePositive} = 1
    options.safetyIterationCap (1,1) double {mustBeInteger,mustBePositive} = 1000000
    options.seed (1,1) double {mustBeInteger,mustBeNonnegative} = 230814
    options.displayEvery (1,1) double {mustBeInteger,mustBeNonnegative} = 0
    options.resume (1,1) logical = true
    options.quickTest (1,1) logical = false
end

folder=fileparts(mfilename('fullpath'));
root=fileparts(folder);
addpath(root,fullfile(root,'nnoe'),fullfile(root,'native'));

if options.quickTest
    options.trainingSamples=min(options.trainingSamples,600);
    options.validationSamples=min(options.validationSamples,300);
    options.orders=3;
    options.architectures={[8 8]};
    options.gains=10;
    options.regularizations=1e-3;
    options.stepSizes=[2e-3 1e-3];
    options.tauProbeIterations=min(options.tauProbeIterations,4);
    options.numInitializations=1;
    options.safetyIterationCap=20;
    fitCheckEvery=1;
    fitPatience=2;
    fitMinDelta=1e9;
    minimumIterations=2;
    suffix='_smoke_test_v2';
else
    fitCheckEvery=25;
    fitPatience=20;
    fitMinDelta=1e-3;
    minimumIterations=500;
    suffix='';
end

for i=1:numel(options.architectures),validate_hidden(options.architectures{i});end
settings=make_ordered_settings(options.orders,options.architectures, ...
    options.gains,options.regularizations);
fprintf(['Campaign: %d primary settings x %d initialization(s) = %d full fits; ' ...
    '%d tau probes per fit.\n'],height(settings),options.numInitializations, ...
    height(settings)*options.numInitializations,numel(options.stepSizes));

out=fullfile(folder,'results',['smart_search_12000' suffix]);
modelsOut=fullfile(out,'models');
if ~exist(modelsOut,'dir'),mkdir(modelsOut);end
probeFile=fullfile(out,'tau_probes.csv');
attemptFile=fullfile(out,'full_attempts.csv');
resultFile=fullfile(out,'model_results.csv');

%% Reproducible data and strict train/validation/test separation.
d=load(fullfile(folder,'data','boucwen_data.mat'));
needed=options.trainingSamples+options.validationSamples;
assert(numel(d.u)>=needed && numel(d.y)>=needed,'Example2:DataLength', ...
    'The generated record must contain at least %d samples.',needed);
u=d.u(:); cleanY=d.y(:);
rng(options.seed,'twister');
noiseStd=8e-3*1e-3; % 8e-3 mm in the raw benchmark unit (m).
noise=noiseStd*randn(needed,1);
measuredY=cleanY(1:needed)+noise;
nTrain=options.trainingSamples;
uTrain=u(1:nTrain); yTrain=measuredY(1:nTrain);
uValidation=u(nTrain+1:needed); yValidation=measuredY(nTrain+1:needed);
scale=1/std(yTrain);

testFolder=fullfile(folder,'BenchmarkFiles','Test signals','Validation signals');
tu=load(fullfile(testFolder,'uval_multisine.mat'));
ty=load(fullfile(testFolder,'yval_multisine.mat'));
uTest=tu.uval_multisine(:); yTest=ty.yval_multisine(:);
metadata=struct('trainingSamples',nTrain,'validationSamples',options.validationSamples, ...
    'noiseStdDataUnits',noiseStd,'noiseStdMm',8e-3,'noiseSeed',options.seed, ...
    'orderedSettings',settings,'stepSizes',options.stepSizes, ...
    'tauProbeIterations',options.tauProbeIterations, ...
    'numInitializations',options.numInitializations, ...
    'safetyIterationCap',options.safetyIterationCap, ...
    'fitCheckEvery',fitCheckEvery,'fitPatience',fitPatience, ...
    'fitMinDelta',fitMinDelta,'minimumIterations',minimumIterations, ...
    'selectionRule','largest validation BFR; test reported for every model');
save(fullfile(out,'search_data.mat'),'uTrain','yTrain','uValidation', ...
    'yValidation','uTest','yTest','scale','noise','metadata');

probeRows=read_or_empty(probeFile,@empty_probe_table,options.resume);
attemptRows=read_or_empty(attemptFile,@empty_attempt_table,options.resume);
resultRows=read_or_empty(resultFile,@empty_result_table,options.resume);

%% Ordered primary fits, each with an initialization-specific tau probe.
for is=1:height(settings)
    setting=settings(is,:);
    hidden=parse_architecture(setting.architecture);
    model=struct('order',setting.order,'hidden',hidden);
    for init=1:options.numInitializations
        completed=resultRows.settingId==setting.settingId & ...
            resultRows.initialization==init & resultRows.success;
        if any(completed)
            fprintf('Skipped completed setting %d/%d, initialization %d.\n', ...
                setting.settingId,height(settings),init);
            continue
        end
        % The seed deliberately excludes K, lambda and tau. Thus all nearby
        % hyperparameters for a structure see the same initialization, and
        % every tau probe belonging to a fit starts from exactly the same x0.
        runSeed=options.seed+100000*setting.structureId+init;
        fprintf(['[%d/%d] n=%d, %s, K=%g, lambda=%g, init=%d: ' ...
            'probing tau...\n'],setting.settingId,height(settings),setting.order, ...
            setting.architecture,setting.gain,setting.regularization,init);

        for it=1:numel(options.stepSizes)
            tau=options.stepSizes(it);
            already=probeRows.settingId==setting.settingId & ...
                probeRows.initialization==init & same_number(probeRows.tau,tau);
            if any(already),continue;end
            started=tic;
            try
                r=identify_nnoe(uTrain,scale*yTrain,model,stepSize=tau, ...
                    gain=setting.gain,regularization=setting.regularization, ...
                    maxIterations=options.tauProbeIterations,tolerance=0, ...
                    displayEvery=options.displayEvery,seed=runSeed, ...
                    fitCheckEvery=options.tauProbeIterations,fitPatience=2, ...
                    fitMinDelta=0,minimumIterations=options.tauProbeIterations, ...
                    restoreBestFit=false);
                maxConstraint=max(r.constraintNorm,[],'omitnan');
                maxVelocity=max(r.velocityNorm,[],'omitnan');
                stable=all(isfinite(r.x)) && isfinite(r.trainingFit) && ...
                    isfinite(maxConstraint) && isfinite(maxVelocity);
                if stable,errorMessage="";else,errorMessage="non-finite probe";end
                trainingBFR=r.trainingFit; elapsed=r.elapsedTime;
                iterations=r.iterations; stopReason=string(r.stopReason);
            catch ME
                stable=false; trainingBFR=NaN; maxConstraint=NaN; maxVelocity=NaN;
                elapsed=toc(started); iterations=0; stopReason="failed";
                errorMessage=string(ME.message);
            end
            row=table(setting.settingId,setting.structureId,setting.order, ...
                string(setting.architecture),setting.parameters,setting.gain, ...
                setting.regularization,tau,init,runSeed,trainingBFR,maxConstraint, ...
                maxVelocity,elapsed,iterations,stopReason,stable,errorMessage, ...
                'VariableNames',probeRows.Properties.VariableNames);
            probeRows=[probeRows;row]; %#ok<AGROW>
            writetable(probeRows,probeFile);
            fprintf('  tau=%g: training BFR %.3f%%, %s.\n',tau,trainingBFR,errorMessage);
        end

        probes=probeRows(probeRows.settingId==setting.settingId & ...
            probeRows.initialization==init & probeRows.stable,:);
        [~,rank]=sort(probes.trainingBFR,'descend','MissingPlacement','last');
        probes=probes(rank,:);
        if isempty(probes)
            warning('Example2:NoStableTau','No stable tau for setting %d.',setting.settingId);
        end

        % Restart from the common initialization. If the leading tau later
        % diverges, continue down the probe ranking rather than losing the fit.
        for ip=1:height(probes)
            tau=probes.tau(ip);
            tried=attemptRows.settingId==setting.settingId & ...
                attemptRows.initialization==init & same_number(attemptRows.tau,tau);
            if any(tried),continue;end
            modelFile=fullfile(modelsOut,sprintf('setting_%02d_init_%02d.mat', ...
                setting.settingId,init));
            started=tic;
            try
                r=identify_nnoe(uTrain,scale*yTrain,model,stepSize=tau, ...
                    gain=setting.gain,regularization=setting.regularization, ...
                    maxIterations=options.safetyIterationCap,tolerance=0, ...
                    displayEvery=options.displayEvery,seed=runSeed, ...
                    fitCheckEvery=fitCheckEvery,fitPatience=fitPatience, ...
                    fitMinDelta=fitMinDelta,minimumIterations=minimumIterations, ...
                    restoreBestFit=true);
                pValidation=simulate_nnoe(r.Net,uValidation, ...
                    scale*yValidation(1:setting.order))/scale;
                [validationRMSE,validationBFR]=nnoe_metrics(yValidation,pValidation);
                % Explicitly evaluate every completed model on the benchmark
                % test record. These values never participate in selection.
                pTest=simulate_nnoe(r.Net,uTest,scale*yTest(1:setting.order))/scale;
                [testRMSE,testBFR]=nnoe_metrics(yTest,pTest);
                fatigue=strcmp(string(r.stopReason),"training FIT stabilized");
                finiteMetrics=all(isfinite([r.trainingFit validationRMSE ...
                    validationBFR testRMSE testBFR])) && all(isfinite(r.x));
                successful=fatigue && finiteMetrics;
                if successful
                    errorMessage="";
                    candidate=struct('Net',r.Net,'result',r, ...
                        'validationPrediction',pValidation,'testPrediction',pTest, ...
                        'validationRMSE',validationRMSE,'validationBFR',validationBFR, ...
                        'testRMSE',testRMSE,'testBFR',testBFR, ...
                        'setting',setting,'initialization',init,'runSeed',runSeed,'tau',tau);
                    save(modelFile,'candidate','metadata','scale','-v7.3');
                elseif ~finiteMetrics
                    errorMessage="non-finite full-fit metric";
                else
                    errorMessage="full fit did not terminate by FIT fatigue";
                end
                trainingBFR=r.trainingFit; elapsed=r.elapsedTime;
                iterations=r.iterations; stopReason=string(r.stopReason);
            catch ME
                successful=false; trainingBFR=NaN; validationRMSE=NaN;
                validationBFR=NaN; testRMSE=NaN; testBFR=NaN;
                elapsed=toc(started); iterations=0; stopReason="failed";
                errorMessage=string(ME.message);
            end
            arow=table(setting.settingId,setting.structureId,setting.order, ...
                string(setting.architecture),setting.parameters,setting.gain, ...
                setting.regularization,tau,init,runSeed,ip,trainingBFR, ...
                validationRMSE,validationBFR,testRMSE,testBFR,elapsed,iterations, ...
                stopReason,successful,errorMessage,'VariableNames', ...
                attemptRows.Properties.VariableNames);
            attemptRows=[attemptRows;arow]; %#ok<AGROW>
            writetable(attemptRows,attemptFile);
            fprintf(['  full tau=%g: train %.3f%%, validation %.3f%%, test %.3f%%, ' ...
                '%d iterations, %.1f s, %s.\n'],tau,trainingBFR,validationBFR, ...
                testBFR,iterations,elapsed,errorMessage);
            if successful
                rrow=table(setting.settingId,setting.structureId,setting.order, ...
                    string(setting.architecture),setting.parameters,setting.gain, ...
                    setting.regularization,tau,init,runSeed,trainingBFR, ...
                    validationRMSE,validationBFR,testRMSE,testBFR,elapsed,iterations, ...
                    stopReason,true,string(modelFile),'VariableNames', ...
                    resultRows.Properties.VariableNames);
                resultRows=[resultRows;rrow]; %#ok<AGROW>
                writetable(resultRows,resultFile);
                break
            end
        end
    end
end

%% Validation-only ranking, with test metrics shown for every model.
campaignSummary=summarize_results(settings,resultRows,options.numInitializations);
writetable(campaignSummary,fullfile(out,'campaign_summary.csv'));
eligible=campaignSummary.numSuccessful>0;
assert(any(eligible),'Example2:NoSuccessfulModel','No primary model completed successfully.');
eligibleRows=find(eligible);
[~,bestLocal]=max(campaignSummary.meanValidationBFR(eligible));
winner=campaignSummary(eligibleRows(bestLocal),:);
winnerRuns=resultRows(resultRows.settingId==winner.settingId,:);
[~,bestInit]=max(winnerRuns.validationBFR);
selectedRun=winnerRuns(bestInit,:);
copy=load(selectedRun.modelFile,'candidate');
selectedModel=copy.candidate;
save(fullfile(out,'selected_model.mat'),'selectedModel','winner', ...
    'selectedRun','metadata','-v7.3');
summary=struct('selectedSetting',winner,'selectedRun',selectedRun, ...
    'modelResults',resultRows,'fullAttempts',attemptRows,'tauProbes',probeRows, ...
    'campaignSummary',campaignSummary,'metadata',metadata);
save(fullfile(out,'campaign_summary.mat'),'summary','-v7.3');
fprintf(['SELECTED BY VALIDATION: n=%d %s, K=%g, lambda=%g, tau=%g; ' ...
    'validation BFR %.3f%%, test BFR %.3f%%.\n'],selectedRun.order, ...
    selectedRun.architecture,selectedRun.gain,selectedRun.regularization, ...
    selectedRun.tau,selectedRun.validationBFR,selectedRun.testBFR);
end

function settings=make_ordered_settings(orders,architectures,gains,lambdas)
% Start with the neighborhoods of the submitted/revised best n=3 models,
% then increase order, and leave the larger 10-by-10 alternatives last.
priorityOrders=[3 3 4 4 5 5 3 4 5];
priorityArchitectures={[8 8],[6 6],[8 8],[6 6],[8 8],[6 6], ...
    [10 10],[10 10],[10 10]};
pairs=cell(0,2);
for i=1:numel(priorityOrders)
    if ismember(priorityOrders(i),orders) && has_architecture(architectures,priorityArchitectures{i})
        pairs(end+1,:)={priorityOrders(i),priorityArchitectures{i}}; %#ok<AGROW>
    end
end
for io=1:numel(orders)
    for ia=1:numel(architectures)
        seen=false;
        for i=1:size(pairs,1)
            seen=seen || (pairs{i,1}==orders(io) && isequal(pairs{i,2},architectures{ia}));
        end
        if ~seen,pairs(end+1,:)={orders(io),architectures{ia}};end %#ok<AGROW>
    end
end
settings=table('Size',[0 9],'VariableTypes',{'double','double','double','string', ...
    'double','double','double','double','double'},'VariableNames', ...
    {'settingId','structureId','order','architecture','numLayers','parameters', ...
    'gain','regularization','priority'});
settingId=0;
for ip=1:size(pairs,1)
    order=pairs{ip,1}; hidden=pairs{ip,2};
    [~,parameterCount]=initialize_nnoe(order,hidden,1,1);
    for il=1:numel(lambdas)
        for ik=1:numel(gains)
            settingId=settingId+1;
            values={settingId,ip,order,string(architecture_label(hidden)), ...
                numel(hidden),parameterCount,gains(ik),lambdas(il),settingId};
            settings=[settings;cell2table(values,'VariableNames', ...
                settings.Properties.VariableNames)]; %#ok<AGROW>
        end
    end
end
end

function T=empty_probe_table
T=table('Size',[0 18],'VariableTypes',{'double','double','double','string', ...
    'double','double','double','double','double','double','double','double', ...
    'double','double','double','string','logical','string'},'VariableNames', ...
    {'settingId','structureId','order','architecture','parameters','gain', ...
    'regularization','tau','initialization','runSeed','trainingBFR', ...
    'maxConstraint','maxVelocity','trainingSeconds','iterations','stopReason', ...
    'stable','errorMessage'});
end

function T=empty_attempt_table
T=table('Size',[0 21],'VariableTypes',{'double','double','double','string', ...
    'double','double','double','double','double','double','double','double', ...
    'double','double','double','double','double','double','string','logical','string'}, ...
    'VariableNames',{'settingId','structureId','order','architecture','parameters', ...
    'gain','regularization','tau','initialization','runSeed','tauRank','trainingBFR', ...
    'validationRMSE','validationBFR','testRMSE','testBFR','trainingSeconds', ...
    'iterations','stopReason','success','errorMessage'});
end

function T=empty_result_table
T=table('Size',[0 20],'VariableTypes',{'double','double','double','string', ...
    'double','double','double','double','double','double','double','double', ...
    'double','double','double','double','double','string','logical','string'}, ...
    'VariableNames',{'settingId','structureId','order','architecture','parameters', ...
    'gain','regularization','tau','initialization','runSeed','trainingBFR', ...
    'validationRMSE','validationBFR','testRMSE','testBFR','trainingSeconds', ...
    'iterations','stopReason','success','modelFile'});
end

function T=read_or_empty(path,constructor,resume)
if resume && isfile(path)
    io=detectImportOptions(path,'TextType','string');
    stringNames=intersect({'architecture','stopReason','errorMessage','modelFile'}, ...
        io.VariableNames,'stable');
    if ~isempty(stringNames),io=setvartype(io,stringNames,'string');end
    T=readtable(path,io);
else
    T=constructor();
end
end

function S=summarize_results(settings,results,required)
S=table('Size',[0 17],'VariableTypes',{'double','double','double','string', ...
    'double','double','double','double','double','double','double','double', ...
    'double','double','double','double','double'},'VariableNames', ...
    {'settingId','structureId','order','architecture','parameters','gain', ...
    'regularization','numSuccessful','requiredInitializations','meanTau', ...
    'meanIterations','meanTrainingSeconds','meanTrainingBFR', ...
    'meanValidationBFR','stdValidationBFR','meanTestBFR','stdTestBFR'});
for i=1:height(settings)
    base=settings(i,:); q=results(results.settingId==base.settingId & results.success,:);
    if isempty(q)
        stats={0,required,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN};
    else
        stats={height(q),required,mean(q.tau),mean(q.iterations), ...
            mean(q.trainingSeconds),mean(q.trainingBFR),mean(q.validationBFR), ...
            std(q.validationBFR),mean(q.testBFR),std(q.testBFR)};
    end
    values=[{base.settingId,base.structureId,base.order,base.architecture, ...
        base.parameters,base.gain,base.regularization},stats];
    S=[S;cell2table(values,'VariableNames',S.Properties.VariableNames)]; %#ok<AGROW>
end
end

function yes=has_architecture(architectures,target)
yes=false;
for i=1:numel(architectures),yes=yes || isequal(architectures{i},target);end
end

function label=architecture_label(hidden)
label=char(join(string(hidden),'-'));
end

function hidden=parse_architecture(label)
hidden=str2double(split(string(label),'-')).';
end

function validate_hidden(hidden)
assert(isnumeric(hidden) && isvector(hidden) && all(isfinite(hidden)) && ...
    all(hidden>0) && all(mod(hidden,1)==0),'Example2:Architecture', ...
    'Each architecture must be a vector of positive integer widths.');
end

function yes=same_number(values,target)
yes=abs(values-target)<=10*eps(max(1,abs(target)));
end
