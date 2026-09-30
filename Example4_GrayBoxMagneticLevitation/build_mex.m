%% Build the packed-Jacobian provider for the current MATLAB architecture.
folder=fileparts(mfilename('fullpath'));
mex('-R2018a','-output',fullfile(folder,'levitator_constraints_mex'), ...
    fullfile(folder,'levitator_constraints_mex.c'));
fprintf('Built levitator_constraints_mex.%s\n',mexext);
