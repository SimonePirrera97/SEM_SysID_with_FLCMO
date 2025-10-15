# System identification based on Simulation Error Minimization (SEM) through FL-CMO optimization

Code implementing the system identification method based on simulation error minimization (SEM) and constrained optimization, as proposed in the paper "A constrained optimization approach to nonlinear system identification through simulation-error minimization" by Vito Cerone, Sophie M. Fosson, Simone Pirrera, and Diego Regruto. 

The main functions consider the case of SEM problems when the model structure is defined by neural NIO models, i.e., NNOE neural networks training. It was used to produce the numerical examples 1-3 in the paper. 
Additionally, we include the code for the gray-box identification task of example 4, which exploits a different definition of the problem constraints.
