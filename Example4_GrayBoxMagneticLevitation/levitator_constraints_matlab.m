function [constraints,jacobian] = levitator_constraints_matlab(x,input,Ts,mass,gravity)
%LEVITATOR_CONSTRAINTS_MATLAB Gray-box equations with a packed Jacobian.
magneticGain=x(1);
offset=x(2);
friction=x(3);
output=x(4:end);
sampleCount=numel(output);
rows=sampleCount-2;

acceleration=(output(3:end)-2*output(2:end-1)+output(1:end-2))/Ts^2;
velocity=(output(2:end-1)-output(1:end-2))/Ts;
constraints=mass*output(1:end-2).^2.*(acceleration-gravity) ...
    + magneticGain*input(1:rows).^2+offset ...
    + friction*output(1:end-2).^2.*velocity;

parameterBlock=[input(1:rows).'.^2;ones(1,rows); ...
    (output(1:rows).^2.*velocity).'];
bandBlock=zeros(3,rows);
y0=output(1:end-2);y1=output(2:end-1);
bandBlock(1,:)=(2*mass*y0.*(acceleration-gravity) ...
    +(mass/Ts^2)*y0.^2 ...
    +friction*(2*y0.*velocity-y0.^2/Ts)).';
bandBlock(2,:)=(-2*(mass/Ts^2)*y0.^2+friction*y0.^2/Ts).';
bandBlock(3,:)=((mass/Ts^2)*y0.^2).';
jacobian=struct('N',sampleCount,'n',2,'p',1,'q',0,'ntheta',3, ...
    'parameterBlock',parameterBlock,'bandBlock',bandBlock);
end
