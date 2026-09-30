function result = flcmo_optimize(x0,objectiveGradient,constraintFunction,options)
%FLCMO_OPTIMIZE Integrate the generic FL-CMO flow using packed Jacobians.
arguments
    x0 double
    objectiveGradient (1,1) function_handle
    constraintFunction (1,1) function_handle
    options.stepSize (1,1) double {mustBePositive} = 1e-3
    options.maxIterations (1,1) double {mustBeInteger,mustBePositive} = 1000
    options.tolerance (1,1) double {mustBeNonnegative} = 1e-4
    options.gain (1,1) double = 1
    options.displayEvery (1,1) double {mustBeInteger,mustBeNonnegative} = 0
    options.storeTrajectory (1,1) logical = false
    options.monitorFunction = []
    options.monitorEvery (1,1) double {mustBeInteger,mustBePositive} = 10
    options.monitorPatience (1,1) double {mustBeInteger,mustBePositive} = 10
    options.monitorMinDelta (1,1) double {mustBeNonnegative} = 1e-2
    options.minimumIterations (1,1) double {mustBeInteger,mustBeNonnegative} = 100
    options.maxElapsedTime (1,1) double {mustBePositive} = Inf
    options.restoreBestMonitor (1,1) logical = false
end

x=x0(:);
if options.storeTrajectory
    trajectory=zeros(numel(x),options.maxIterations+1);
    trajectory(:,1)=x;
else
    trajectory=[];
end
constraintNorm=zeros(options.maxIterations,1);
velocityNorm=zeros(options.maxIterations,1);
kernelTime=zeros(options.maxIterations,1);
startTime=tic;
monitorHistory=zeros(ceil(options.maxIterations/options.monitorEvery),2);
monitorCount=0; bestMonitor=-inf; bestMonitorIteration=0; bestMonitorX=[];
staleChecks=0; stopReason="maximum iterations";

for iteration=1:options.maxIterations
    [velocity,constraints,stats]=flcmo_step( ...
        x,objectiveGradient,constraintFunction,options.gain);
    velocityNorm(iteration)=norm(velocity,inf);
    constraintNorm(iteration)=norm(constraints,inf);
    kernelTime(iteration)=stats.factorizationTime+stats.solveTime;
    x=x+options.stepSize*velocity;
    if options.storeTrajectory,trajectory(:,iteration+1)=x;end

    if options.displayEvery>0 && mod(iteration,options.displayEvery)==0
        fprintf('iter %d: ||h||_inf=%.3e, ||xdot||_inf=%.3e\n', ...
            iteration,constraintNorm(iteration),velocityNorm(iteration));
    end
    if max(velocityNorm(iteration),constraintNorm(iteration))<=options.tolerance
        stopReason="numerical tolerance";
        break
    end
    if toc(startTime)>=options.maxElapsedTime
        stopReason="wall-time limit";
        break
    end
    if ~isempty(options.monitorFunction) && mod(iteration,options.monitorEvery)==0
        monitorCount=monitorCount+1; value=options.monitorFunction(x);
        monitorHistory(monitorCount,:)=[iteration value];
        if value>bestMonitor+options.monitorMinDelta
            bestMonitor=value; bestMonitorIteration=iteration; bestMonitorX=x;
            staleChecks=0;
        else
            staleChecks=staleChecks+1;
        end
        if iteration>=options.minimumIterations && staleChecks>=options.monitorPatience
            stopReason="training FIT stabilized";
            break
        end
    end
end

if options.restoreBestMonitor && ~isempty(bestMonitorX)
    x=bestMonitorX;
end

result.x=x;
result.iterations=iteration;
result.bestMonitor=bestMonitor;
result.bestMonitorIteration=bestMonitorIteration;
result.restoredBestMonitor=options.restoreBestMonitor && ~isempty(bestMonitorX);
result.elapsedTime=toc(startTime);
result.constraintNorm=constraintNorm(1:iteration);
result.velocityNorm=velocityNorm(1:iteration);
result.kernelTime=kernelTime(1:iteration);
result.monitorHistory=monitorHistory(1:monitorCount,:);
result.stopReason=stopReason;
if options.storeTrajectory
    result.trajectory=trajectory(:,1:iteration+1);
else
    result.trajectory=[];
end
end
