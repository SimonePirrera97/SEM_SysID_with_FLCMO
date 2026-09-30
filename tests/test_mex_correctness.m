function report = test_mex_correctness
%TEST_MEX_CORRECTNESS Compare MATLAB and MEX NNOE constraint structures.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'nnoe'),fullfile(root,'native'));
rng(7);
order=2;hidden=[4 3];sampleCount=24;outputDimension=2;inputDimension=2;
[~,parameterCount]=initialize_nnoe(order,hidden,inputDimension,outputDimension);
theta=0.1*randn(parameterCount,1);
output=0.2*randn(sampleCount,outputDimension);
input=0.2*randn(sampleCount,inputDimension);

start=tic;[constraintsMatlab,jacobianMatlab]=nnoe_constraints_matlab( ...
    theta,output,input,order,hidden);report.matlabTime=toc(start);
start=tic;[constraintsMex,jacobianMex]=nnoe_constraints_mex( ...
    theta,output,input,order,hidden);report.mexTime=toc(start);
report.constraintRelativeError=relative_error(constraintsMex,constraintsMatlab);
report.parameterBlockRelativeError=relative_error( ...
    jacobianMex.parameterBlock,jacobianMatlab.parameterBlock);
report.bandBlockRelativeError=relative_error( ...
    jacobianMex.bandBlock,jacobianMatlab.bandBlock);

% Verify the complete native projection against an explicitly expanded J.
gradient=randn(parameterCount+outputDimension*sampleCount,1);gain=1.7;
nativeVelocity=sidqr_mex(jacobianMex,gradient,constraintsMex,gain);
J=expand_jacobian(jacobianMatlab);
rhs=-J*gradient+gain*constraintsMatlab;
multipliers=(J*J')\rhs;
referenceVelocity=-(gradient+J'*multipliers);
report.velocityRelativeError=relative_error(nativeVelocity,referenceVelocity);
disp(struct2table(report));
errors=[report.constraintRelativeError,report.parameterBlockRelativeError, ...
    report.bandBlockRelativeError,report.velocityRelativeError];
assert(max(errors)<1e-11);
end

function J=expand_jacobian(packed)
constraintCount=packed.p*(packed.N-packed.n);
bandHeight=(packed.n+1)*(packed.p+packed.q);
entriesPerColumn=packed.ntheta+bandHeight;
rowIndices=repmat(1:constraintCount,entriesPerColumn,1);
parameterIndices=repmat((1:packed.ntheta).',1,constraintCount);
bandIndices=zeros(bandHeight,constraintCount);
for column=1:constraintCount
    block=floor((column-1)/packed.p);
    bandIndices(:,column)=packed.ntheta+(packed.p+packed.q)*block+(1:bandHeight).';
end
columnIndices=[parameterIndices;bandIndices];
values=[packed.parameterBlock;packed.bandBlock];
J=sparse(rowIndices(:),columnIndices(:),values(:),constraintCount, ...
    packed.ntheta+(packed.p+packed.q)*packed.N);
end

function value=relative_error(actual,reference)
value=norm(actual-reference,'fro')/max(1,norm(reference,'fro'));
end
