function run_cmo_replicates
% Four additional initializations for five retained FL-CMO architectures.
folder=fileparts(mfilename('fullpath')); root=fileparts(folder);
addpath(root,fullfile(root,'nnoe'),fullfile(root,'native'));
out=fullfile(folder,'results','replicates'); if ~exist(out,'dir'),mkdir(out);end
tr=load(fullfile(folder,'data','dataset_training.mat'));
te=load(fullfile(folder,'data','dataset_test.mat'));
u=tr.utrain(1:3500,:); y=tr.ytrain(1:3500,1:2);
ut=te.u_test(1:5000,:); yt=te.y_test(1:5000,1:2);
architectures={10,15,[5 5],[7 7],[5 5 5]}; paperRows=[3 4 6 7 10];
stepSize=1e-2; regularization=1e-3; gain=1;
path=fullfile(out,'cmo_replicates.csv'); rows=cell(0,25);
if exist(path,'file'), old=readtable(path,'TextType','string'); else, old=table; end
for replicate=1:4
    for j=1:numel(architectures)
        seed=1000*replicate+paperRows(j);
        if ~isempty(old) && any(old.replicate==replicate & old.paperRow==paperRows(j)),continue;end
        model=struct('order',3,'hidden',architectures{j});
        r=identify_nnoe(u,y,model,stepSize=stepSize,gain=gain, ...
            regularization=regularization,maxIterations=10000,tolerance=1e-4, ...
            displayEvery=0,seed=seed,fitCheckEvery=10,fitPatience=10, ...
            fitMinDelta=1e-2,minimumIterations=200,restoreBestFit=true);
        if strcmp(r.stopReason,'maximum iterations')
            error('Example3:NoFatigue','Replicate %d row %d did not fatigue.',replicate,paperRows(j));
        end
        pTrain=simulate_nnoe(r.Net,u,y(1:3,:)); [trm,trb]=nnoe_metrics(y,pTrain);
        p=simulate_nnoe(r.Net,ut,yt(1:3,:)); [rm,bfr]=nnoe_metrics(yt,p);
        label=char(strjoin(string(architectures{j}),'x'));
        rows(end+1,:)={"FL-CMO",replicate,seed,paperRows(j),label,numel(architectures{j}), ...
            r.Net.numParams,trm(1),trm(2),trb(1),trb(2),rm(1),rm(2),bfr(1),bfr(2), ...
            mean(bfr),r.elapsedTime,r.iterations,r.elapsedTime/r.iterations,stepSize,gain, ...
            regularization,10,10,char(r.stopReason)}; %#ok<AGROW>
        vars={'method','replicate','initializationSeed','paperRow','hidden','layers', ...
            'parameters','trainingRMSE1','trainingRMSE2','trainingBFR1','trainingBFR2', ...
            'testRMSE1','testRMSE2','testBFR1','testBFR2','meanTestBFR','trainingSeconds', ...
            'iterations','secondsPerIteration','stepSize','gain','regularization', ...
            'monitorEvery','fitPatience','stopReason'};
        fresh=cell2table(rows,'VariableNames',vars);
        if isempty(old),combined=fresh;else,combined=[old;fresh];end
        writetable(combined,path);
        save(fullfile(out,sprintf('cmo_rep%d_row_%02d.mat',replicate,paperRows(j))), ...
            'r','p','pTrain','model','replicate','seed','stepSize','gain','regularization');
        fprintf('CMO replicate %d row %d: %.2f/%.2f%%, %.2fs.\n', ...
            replicate,paperRows(j),bfr(1),bfr(2),r.elapsedTime);
    end
end
end
