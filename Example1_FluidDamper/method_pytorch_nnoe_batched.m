function runs = method_pytorch_nnoe_batched(cfg)
%METHOD_PYTORCH_NNOE_BATCHED Truncated NNOE with length-32/64/128 sequences.
py=fullfile(cfg.root,'.venv','bin','python'); if ~exist(py,'file'), py='python3'; end
script=fullfile(cfg.folder,'run_pytorch_es1.py');
cmd=sprintf('"%s" "%s" --method nnoe_batched --data "%s" --initializations "%s" --out "%s" --timeout-seconds %.15g --runs %d --log "%s"', ...
    py,script,fullfile(cfg.folder,'data','data_es2.mat'),cfg.initializationFile,cfg.work,cfg.fitTimeoutSeconds,cfg.runs,cfg.logFile);
[status,text]=system(cmd); assert(status==0,'PyTorch batched NNOE failed:\n%s',text);
fprintf('%s',text); runs=readtable(fullfile(cfg.work,'nnoe_batched_pytorch_runs.csv'));
es1_log(cfg,'Loaded PyTorch NNOE_BATCHED results',sprintf('%d run rows loaded',height(runs)));
end
