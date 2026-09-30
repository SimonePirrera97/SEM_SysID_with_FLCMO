function result = identify_nnoe(input,output,model,options)
%IDENTIFY_NNOE Identify an NNOE using a selectable packed constraint function.
%
% constraintFunction can name either implementation:
%   @nnoe_constraints_mex       optimized native evaluation (default)
%   @nnoe_constraints_matlab    readable MATLAB reference
arguments
    input double
    output double
    model struct
    options.constraintFunction (1,1) function_handle = @nnoe_constraints_mex
    options.stepSize (1,1) double {mustBePositive} = 2e-3
    options.maxIterations (1,1) double {mustBeInteger,mustBePositive} = 1000
    options.gain (1,1) double = 1
    options.regularization (1,1) double {mustBeNonnegative} = 1e-3
    options.tolerance (1,1) double {mustBeNonnegative} = 1e-3
    options.displayEvery (1,1) double {mustBeInteger,mustBeNonnegative} = 100
    options.x0 double = []
    options.seed (1,1) double = 0
    options.storeTrajectory (1,1) logical = false
    options.fitCheckEvery (1,1) double {mustBeInteger,mustBePositive} = 10
    options.fitPatience (1,1) double {mustBeInteger,mustBePositive} = 10
    options.fitMinDelta (1,1) double {mustBeNonnegative} = 1e-2
    options.minimumIterations (1,1) double {mustBeInteger,mustBeNonnegative} = 100
    options.maxElapsedTime (1,1) double {mustBePositive} = Inf
    options.restoreBestFit (1,1) logical = false
end
assert(isfield(model,'order')&&isfield(model,'hidden'), ...
    'identify_nnoe:Model','model requires order and hidden fields.');
assert(size(input,1)==size(output,1),'identify_nnoe:Data', ...
    'Input and output must contain the same number of samples.');

rng(options.seed);
[~,parameterCount]=initialize_nnoe( ...
    model.order,model.hidden,size(input,2),size(output,2));
sampleCount=size(output,1);
outputDimension=size(output,2);
targetVector=reshape(output.',[],1);

if isempty(options.x0)
    x0=0.01*randn(parameterCount+numel(output),1);
else
    x0=options.x0(:);
    assert(numel(x0)==parameterCount+numel(output), ...
        'identify_nnoe:InitialPoint','x0 has the wrong number of elements.');
end

objectiveGradient=@(x)[2*options.regularization*x(1:parameterCount); ...
    2*(x(parameterCount+1:end)-targetVector)];
constraintFunction=@(x)options.constraintFunction( ...
    x(1:parameterCount), ...
    reshape(x(parameterCount+1:end),outputDimension,sampleCount).', ...
    input,model.order,model.hidden);
fitMonitor=@(x)training_fit(x(1:parameterCount),input,output,model);

optimization=flcmo_optimize(x0,objectiveGradient,constraintFunction, ...
    stepSize=options.stepSize,maxIterations=options.maxIterations, ...
    tolerance=options.tolerance,gain=options.gain, ...
    displayEvery=options.displayEvery,storeTrajectory=options.storeTrajectory, ...
    monitorFunction=fitMonitor,monitorEvery=options.fitCheckEvery, ...
    monitorPatience=options.fitPatience,monitorMinDelta=options.fitMinDelta, ...
    minimumIterations=options.minimumIterations,maxElapsedTime=options.maxElapsedTime, ...
    restoreBestMonitor=options.restoreBestFit);

network=nnoe_from_vector(optimization.x(1:parameterCount),model.order, ...
    model.hidden,size(input,2),outputDimension);
simulatedOutput=simulate_nnoe(network,input,output(1:model.order,:));
trainingFit=100*(1-norm(simulatedOutput-output)/norm(output-mean(output,1)));

result=optimization;
result.Net=network;
result.trainingFit=trainingFit;
result.model=model;
end

function fit=training_fit(parameters,input,output,model)
network=nnoe_from_vector(parameters,model.order,model.hidden,size(input,2),size(output,2));
prediction=simulate_nnoe(network,input,output(1:model.order,:));
fit=100*(1-norm(prediction-output)/norm(output-mean(output,1)));
end
