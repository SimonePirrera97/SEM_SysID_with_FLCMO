function [constraints,jacobian] = nnoe_constraints_matlab(theta,output,input,order,hidden)
%NNOE_CONSTRAINTS_MATLAB MATLAB reference returning the packed Jacobian.
outputDimension=size(output,2);
sampleCount=size(output,1);
network=nnoe_from_vector(theta,order,hidden,size(input,2),outputDimension);
constraints=nnoe_constraint_values(network,output,input);

% The reference derivative is assembled explicitly, then packed into the
% same representation returned directly by nnoe_constraints_mex.
denseJacobian=nnoe_jacobian_matlab(network,output,input);
parameterCount=numel(theta);
constraintCount=outputDimension*(sampleCount-order);
bandHeight=(order+1)*outputDimension;
parameterBlock=denseJacobian(:,1:parameterCount).';
bandBlock=zeros(bandHeight,constraintCount);
for column=1:constraintCount
    outputBlock=floor((column-1)/outputDimension);
    firstSignal=parameterCount+outputBlock*outputDimension+1;
    bandBlock(:,column)=denseJacobian(column,firstSignal:firstSignal+bandHeight-1).';
end
jacobian=struct('N',sampleCount,'n',order,'p',outputDimension,'q',0, ...
    'ntheta',parameterCount,'parameterBlock',parameterBlock,'bandBlock',bandBlock);
end
