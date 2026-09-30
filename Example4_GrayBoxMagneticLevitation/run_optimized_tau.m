function summary = run_optimized_tau(tau, outputSuffix, parameterGuess)
%RUN_OPTIMIZED_TAU Run Example-4 FL-CMO with the parent packed solver.
% The levitator MEX provider supplies the custom packed constraint Jacobian;
% flcmo_optimize and sidqr_mex are the shared parent-folder implementation.
arguments
    tau (1,1) double {mustBePositive}
    outputSuffix (1,1) string = "tau"
    parameterGuess (3,1) double = [2.5e-4;2.5e-5;3e-2]
end

folder=fileparts(mfilename('fullpath'));
root=fileparts(folder);
addpath(root,fullfile(root,'native'),folder);
d=load(fullfile(folder,'data','levitat_data.mat'));
input=d.u(1:1000);
target=d.y(1:1000);
implementation=@levitator_constraints_mex;
objectiveGradient=@(x)[zeros(3,1);2*(x(4:end)-target)];
constraintFunction=@(x)implementation(x,input,d.Ts,d.m,d.g);
validationFunction=@(x)openLoopFit(x(1:3),d.u_validation(:), ...
    d.y_validation_true(:),d.Ts,d.m,d.g);

x0=[parameterGuess;target];
result=flcmo_optimize(x0,objectiveGradient,constraintFunction, ...
    stepSize=tau,maxIterations=100000,gain=1,displayEvery=1000, ...
    tolerance=0,monitorFunction=validationFunction,monitorEvery=100, ...
    minimumIterations=1000,monitorPatience=20,monitorMinDelta=1e-5, ...
    restoreBestMonitor=true);

trueParameters=[d.Km;d.k0;d.c];
estimatedParameters=result.x(1:3);
validationFit=validationFunction(result.x);
summary=struct('tau',tau,'trueParameters',trueParameters, ...
    'estimatedParameters',estimatedParameters,'validationFit',validationFit, ...
    'iterations',result.iterations,'elapsedTime',result.elapsedTime, ...
    'stopReason',result.stopReason,'bestMonitor',result.bestMonitor, ...
    'bestMonitorIteration',result.bestMonitorIteration, ...
    'constraintNorm',result.constraintNorm(end));
fprintf('tau=%.3g | iterations=%d | time=%.6f s | FIT=%.8f | ', ...
    tau,result.iterations,result.elapsedTime,validationFit);
fprintf('theta=[%.9e %.9e %.9e] | stop=%s\n',estimatedParameters,result.stopReason);
save(fullfile(folder,'results','optimized_result_'+outputSuffix+'.mat'), ...
    'result','summary','trueParameters','estimatedParameters','validationFit');
end

function fit=openLoopFit(parameters,input,target,Ts,mass,gravity)
prediction=zeros(size(target));
prediction(1:2)=target(1:2);
for index=1:numel(target)-2
    gap=prediction(index);
    velocity=(prediction(index+1)-gap)/Ts;
    acceleration=gravity-(parameters(1)*input(index)^2+parameters(2)) ...
        /(mass*gap^2)-parameters(3)*velocity/mass;
    prediction(index+2)=2*prediction(index+1)-gap+Ts^2*acceleration;
end
fit=1-sum((prediction-target).^2)/sum((target-mean(target)).^2);
if ~isfinite(fit), fit=-Inf; end
end
