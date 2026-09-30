function results = benchmark_examples(options)
%BENCHMARK_EXAMPLES Compare MATLAB and MEX packed constraint evaluation.
arguments
    options.repetitions (1,1) double {mustBeInteger,mustBePositive} = 5
    options.warmup (1,1) double {mustBeInteger,mustBeNonnegative} = 2
end
folder=fileparts(mfilename('fullpath'));root=fileparts(folder);
addpath(root,fullfile(root,'nnoe'),fullfile(root,'native'));

d=load(fullfile(root,'Example1_FluidDamper','data','data_es2.mat'));
cases(1)=makeCase('Fluid Damper',d.u_train,d.y_train,4,6); %#ok<AGROW>
d=load(fullfile(root,'Example3_MIMOWienerHammerstein','data','dataset_training.mat'));
cases(2)=makeCase('Hammerstein-Wiener',d.utrain(1:3500,:),d.ytrain(1:3500,1:2),3,[8 8]); %#ok<AGROW>

for caseIndex=1:numel(cases)
    testCase=cases(caseIndex);
    for repetition=1:options.warmup
        runStep(testCase,@nnoe_constraints_matlab);
        runStep(testCase,@nnoe_constraints_mex);
    end
    matlabTimes=zeros(options.repetitions,1);mexTimes=matlabTimes;
    for repetition=1:options.repetitions
        start=tic;matlabVelocity=runStep(testCase,@nnoe_constraints_matlab);matlabTimes(repetition)=toc(start);
        start=tic;mexVelocity=runStep(testCase,@nnoe_constraints_mex);mexTimes(repetition)=toc(start);
    end
    results(caseIndex).example=testCase.name;
    results(caseIndex).samples=size(testCase.input,1);
    results(caseIndex).matlabSeconds=median(matlabTimes);
    results(caseIndex).mexSeconds=median(mexTimes);
    results(caseIndex).speedup=results(caseIndex).matlabSeconds/results(caseIndex).mexSeconds;
    results(caseIndex).reductionPercent=100*(1-results(caseIndex).mexSeconds/results(caseIndex).matlabSeconds);
    results(caseIndex).stepRelativeDifference=relativeError(mexVelocity,matlabVelocity);
    assert(results(caseIndex).stepRelativeDifference<1e-10);
end
disp(struct2table(results));
save(fullfile(folder,'benchmark_results.mat'),'results');
writetable(struct2table(results),fullfile(folder,'benchmark_results.csv'));
end

function testCase=makeCase(name,input,output,order,hidden)
[~,parameterCount]=initialize_nnoe(order,hidden,size(input,2),size(output,2));
testCase=struct('name',name,'input',input,'output',output,'order',order, ...
    'hidden',hidden,'parameterCount',parameterCount, ...
    'x0',0.01*randn(parameterCount+numel(output),1));
end

function velocity=runStep(testCase,implementation)
target=reshape(testCase.output.',[],1);
objective=@(x)[2e-3*x(1:testCase.parameterCount); ...
    2*(x(testCase.parameterCount+1:end)-target)];
constraints=@(x)implementation(x(1:testCase.parameterCount), ...
    reshape(x(testCase.parameterCount+1:end),size(testCase.output,2),[]).', ...
    testCase.input,testCase.order,testCase.hidden);
velocity=flcmo_step(testCase.x0,objective,constraints,1);
end

function value=relativeError(actual,reference)
value=norm(actual-reference)/max(1,norm(reference));
end
