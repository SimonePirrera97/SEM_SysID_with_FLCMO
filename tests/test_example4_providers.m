function report = test_example4_providers
%TEST_EXAMPLE4_PROVIDERS Verify interchangeable packed MATLAB/MEX functions.
root=fileparts(fileparts(mfilename('fullpath')));
folder=fullfile(root,'Example4_GrayBoxMagneticLevitation');
addpath(root,fullfile(root,'native'),folder);
d=load(fullfile(folder,'data','levitat_data.mat'));
sampleCount=200;input=d.u(1:sampleCount);target=d.y(1:sampleCount);
rng(11);x0=[2e-4;2e-5;2e-2;target+1e-4*randn(sampleCount,1)];
[constraintsMatlab,jacobianMatlab]=levitator_constraints_matlab( ...
    x0,input,d.Ts,d.m,d.g);
[constraintsMex,jacobianMex]=levitator_constraints_mex(x0,input,d.Ts,d.m,d.g);
report.constraintRelativeError=relativeError(constraintsMex,constraintsMatlab);
report.parameterBlockRelativeError=relativeError( ...
    jacobianMex.parameterBlock,jacobianMatlab.parameterBlock);
report.bandBlockRelativeError=relativeError( ...
    jacobianMex.bandBlock,jacobianMatlab.bandBlock);
direction=randn(size(x0));epsilon=1e-7;
finiteDifference=(levitator_constraints_matlab(x0+epsilon*direction,input,d.Ts,d.m,d.g) ...
    -levitator_constraints_matlab(x0-epsilon*direction,input,d.Ts,d.m,d.g))/(2*epsilon);
analyticDerivative=jacobianMatlab.parameterBlock.'*direction(1:3);
for row=1:sampleCount-2
    analyticDerivative(row)=analyticDerivative(row) ...
        +jacobianMatlab.bandBlock(:,row).'*direction(3+row:5+row);
end
report.directionalDerivativeRelativeError=relativeError( ...
    analyticDerivative,finiteDifference);

objective=@(x)[zeros(3,1);2*(x(4:end)-target)];
matlabFunction=@(x)levitator_constraints_matlab(x,input,d.Ts,d.m,d.g);
mexFunction=@(x)levitator_constraints_mex(x,input,d.Ts,d.m,d.g);
matlabResult=flcmo_optimize(x0,objective,matlabFunction, ...
    maxIterations=20,stepSize=1e-3,gain=1);
mexResult=flcmo_optimize(x0,objective,mexFunction, ...
    maxIterations=20,stepSize=1e-3,gain=1);
report.solutionRelativeDifference=relativeError(mexResult.x,matlabResult.x);
disp(struct2table(report));
assert(max([report.constraintRelativeError,report.parameterBlockRelativeError, ...
    report.bandBlockRelativeError])<1e-12);
assert(report.directionalDerivativeRelativeError<1e-7);
assert(report.solutionRelativeDifference<1e-8);
end

function value=relativeError(actual,reference)
value=norm(actual-reference,'fro')/max(1,norm(reference,'fro'));
end
