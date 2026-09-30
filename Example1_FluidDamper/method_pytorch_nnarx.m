function runs = method_pytorch_nnarx(cfg)
%METHOD_PYTORCH_NNARX Train the one-step-ahead model in PyTorch.
runs=run_python(cfg,'nnarx');
end

function runs=run_python(cfg,method)
py=fullfile(cfg.root,'.venv','bin','python');
if ~exist(py,'file'), py='python3'; end
script=fullfile(cfg.folder,'run_pytorch_es1.py');
cmd=sprintf('"%s" "%s" --method %s --data "%s" --initializations "%s" --out "%s" --timeout-seconds %.15g --runs %d --log "%s"', ...
    py,script,method,fullfile(cfg.folder,'data','data_es2.mat'),cfg.initializationFile,cfg.work,cfg.fitTimeoutSeconds,cfg.runs,cfg.logFile);
[status,text]=system(cmd); assert(status==0,'PyTorch %s failed:\n%s',method,text);
fprintf('%s',text); runs=readtable(fullfile(cfg.work,[method '_pytorch_runs.csv']));
es1_log(cfg,['Loaded PyTorch ' upper(method) ' results'],sprintf('%d run rows loaded',height(runs)));
end
