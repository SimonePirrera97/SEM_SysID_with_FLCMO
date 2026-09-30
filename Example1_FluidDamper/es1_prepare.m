function cfg = es1_prepare(folder)
%ES1_PREPARE Fixed model, data split, and common initial points for Example 1.
cfg.folder = folder;
cfg.root = fileparts(folder);
cfg.out = fullfile(folder,'results');
if ~exist(cfg.out,'dir'), mkdir(cfg.out); end
cfg.work = tempname;
mkdir(cfg.work);
cfg.logFile = fullfile(folder,'results','results.txt');
fid=fopen(cfg.logFile,'w');
assert(fid~=-1,'Cannot initialize log file: %s',cfg.logFile);
fclose(fid);

d = load(fullfile(folder,'data','data_es2.mat'));
cfg.train.u = d.u_train(1:1600,:);
cfg.train.y = d.y_train(1:1600,:);
cfg.validation.u = d.u_train(1601:end,:);
cfg.validation.y = d.y_train(1601:end,:);
cfg.test.u = d.u_valid;
cfg.test.y = d.y_valid;
cfg.order = 4;
cfg.hidden = 8;
cfg.runs = 25;
cfg.nnsysid = getenv('NNSYSID20_PATH');
if isempty(cfg.nnsysid)
    cfg.nnsysid = fullfile(userpath,'SupportPackages','NNSYSID20');
end

% Canonical layout: vec(W1y'), vec(W1u'), b1, W2, b2. This is also the
% native FLCMO parameter ordering; the other methods map it explicitly.
parameterCount = cfg.hidden*(2*cfg.order+1) + cfg.hidden + cfg.hidden + 1;
theta0 = zeros(parameterCount,cfg.runs);
flcmoOutput0 = zeros(size(cfg.train.y,1),cfg.runs);
for run = 1:cfg.runs
    rng(1000+run,'twister');
    theta0(:,run) = 0.01*randn(parameterCount,1);
    % Match identify_nnoe's successful default scale for FLCMO's auxiliary
    % trajectory without changing the common network initialization.
    flcmoOutput0(:,run) = 0.01*randn(size(cfg.train.y,1),1);
end
cfg.initializationFile = fullfile(cfg.work,'common_initializations.mat');
save(cfg.initializationFile,'theta0');
es1_log(cfg,'Prepared common initializations',sprintf('%d initializations saved to %s',cfg.runs,cfg.initializationFile));

es1_log(cfg,'Prepared chronological data split',sprintf('train=%d, validation=%d, test=%d samples',numel(cfg.train.y),numel(cfg.validation.y),numel(cfg.test.y)));
cfg.theta0 = theta0;
cfg.flcmoOutput0 = flcmoOutput0;
end
