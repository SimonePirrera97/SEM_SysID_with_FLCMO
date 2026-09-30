function [velocity,constraints,stats] = flcmo_step(x,objectiveGradient,constraintFunction,gain)
%FLCMO_STEP Evaluate one feedback-linearizing constrained-optimization step.
% The constraint function may be MATLAB or MEX; both return [h,Jstruct].
arguments
    x double
    objectiveGradient (1,1) function_handle
    constraintFunction (1,1) function_handle
    gain (1,1) double = 1
end
gradient=objectiveGradient(x);
[constraints,jacobian]=constraintFunction(x);
[velocity,stats]=sidqr_mex(jacobian,gradient,constraints,gain);
end
