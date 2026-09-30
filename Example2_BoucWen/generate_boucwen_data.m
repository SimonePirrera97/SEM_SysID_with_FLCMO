function outputFile = generate_boucwen_data(seed)
%GENERATE_BOUCWEN_DATA Run the unmodified official benchmark generator.
% The protected Newmark integrator and official script remain in
% BenchmarkFiles. This wrapper fixes the random phase seed and saves their
% noiseless 40,960-sample result under data/. Measurement noise is added by
% main_smart_search after the training/validation split.
arguments
    seed (1,1) double {mustBeInteger,mustBeNonnegative} = 220812
end

folder=fileparts(mfilename('fullpath'));
benchmarkFolder=fullfile(folder,'BenchmarkFiles');
generator=fullfile(benchmarkFolder,'BoucWen_ExampleIntegration.m');
integrator=fullfile(benchmarkFolder,'BoucWen_NewmarkIntegration.p');
assert(isfile(generator) && isfile(integrator),'Example2:BenchmarkFiles', ...
    'The official generator and protected Newmark integrator are required.');

oldRng=rng;
rngCleanup=onCleanup(@()rng(oldRng));
pathCleanup=onCleanup(@()rmpath(benchmarkFolder));
addpath(benchmarkFolder);
rng(seed,'twister');
% The official generator is a script. Predeclarations make its workspace
% contract explicit and let Code Analyzer validate the wrapper.
u=[]; y=[]; fs=NaN; fsint=NaN; upsamp=NaN; P=NaN; N=NaN;
fmin=NaN; fmax=NaN;
run(generator);

assert(isequal(size(u),[40960 1]) && numel(y)==40960, ...
    'Example2:GeneratedSize','Official generator returned unexpected dimensions.');
assert(all(isfinite(u)) && all(isfinite(y)), ...
    'Example2:GeneratedValues','Official generator returned non-finite values.');
y=reshape(y,1,[]);
generation=struct('seed',seed,'samplingFrequencyHz',fs, ...
    'integrationSamplingFrequencyHz',fsint,'upsamplingFactor',upsamp, ...
    'samples',numel(u),'periods',P,'pointsPerPeriod',N, ...
    'inputBandHz',[fmin fmax],'inputStandardDeviation',std(u), ...
    'outputStandardDeviation',std(y), ...
    'generator','BenchmarkFiles/BoucWen_ExampleIntegration.m', ...
    'integrator','BenchmarkFiles/BoucWen_NewmarkIntegration.p');
outputFile=fullfile(folder,'data','boucwen_data.mat');
save(outputFile,'u','y','generation');
fprintf('Saved %d official Bouc--Wen samples to %s.\n',numel(u),outputFile);
end
