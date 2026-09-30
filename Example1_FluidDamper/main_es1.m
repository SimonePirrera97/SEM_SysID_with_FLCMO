function summary = main_es1
%MAIN_ES1 Run all fixed order-4, eight-neuron Example 1 comparisons.
folder=fileparts(mfilename('fullpath')); cfg=es1_prepare(folder);
cleanup=onCleanup(@() rmdir(cfg.work,'s'));
es1_log(cfg,'Started comparison',sprintf('5 methods, %d initializations per method',cfg.runs));
cfg.fitTimeoutSeconds=60;
es1_log(cfg,'Configured uniform fit timeout',sprintf('%.6g seconds for every validation candidate and final fit',cfg.fitTimeoutSeconds));
methods={@method_flcmo,@method_nnsysid_nnoe,@method_pytorch_nnarx,@method_pytorch_nnoe_batched,@method_pytorch_nnoe};
names={'FLCMO','NNSYSID_NNOE','PyTorch_NNARX','PyTorch_NNOE_Batched','PyTorch_NNOE'};
allRuns=cell(size(methods)); rows=zeros(numel(methods),8);
for i=1:numel(methods)
    es1_log(cfg,['Started method ' names{i}],sprintf('%d common initializations',cfg.runs));
    allRuns{i}=methods{i}(cfg);
    fit=allRuns{i}.testFIT; elapsed=allRuns{i}.seconds;
    rows(i,:)=[mean(fit) max(fit) min(fit) std(fit) ...
        mean(elapsed) min(elapsed) max(elapsed) std(elapsed)];
    es1_log(cfg,['Computed statistics for ' names{i}],sprintf('FIT mean=%.6g%%, time mean=%.6g s',rows(i,1),rows(i,5)));
end
summary=array2table(rows,'RowNames',names,'VariableNames', ...
    {'meanFIT','bestFIT','worstFIT','stdFIT','meanSeconds','bestSeconds','worstSeconds','stdSeconds'});
runBlocks=cell(numel(names),1);
for i=1:numel(names)
    block=allRuns{i}(:,{'run','testRMSE','testFIT','seconds'});
    block.Properties.VariableNames{'seconds'}='executionTimeSeconds';
    block.method=repmat(string(names{i}),height(block),1);
    block=movevars(block,'method','Before','run');
    runBlocks{i}=block;
end
allTestRuns=vertcat(runBlocks{:});
writetable(allTestRuns,fullfile(cfg.out,'results.csv'));
writetable(summary,fullfile(cfg.out,'summary.csv'),'WriteRowNames',true);
write_latex_summary(summary,names,fullfile(cfg.out,'summary.tex'));
es1_log(cfg,'Saved aggregate results',sprintf('%d run-level rows written to %s',height(allTestRuns),fullfile(cfg.out,'results.csv')));
es1_log(cfg,'Saved summary tables',sprintf('%s and %s',fullfile(cfg.out,'summary.csv'),fullfile(cfg.out,'summary.tex')));
disp(summary);
es1_log(cfg,'Completed comparison','all requested methods finished successfully');
end

function write_latex_summary(summary,names,path)
fid=fopen(path,'w'); assert(fid~=-1,'Could not create LaTeX table: %s',path);
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'\\begin{tabular}{lrrrrrrrr}\n');
fprintf(fid,'\\hline\n');
fprintf(fid,'Method & FIT avg. & FIT best & FIT worst & FIT std. & Time avg. (s) & Time best (s) & Time worst (s) & Time std. (s) \\\\\n');
fprintf(fid,'\\hline\n');
for i=1:height(summary)
    label=strrep(names{i},'_','\_');
    values=summary{i,:};
    fprintf(fid,'%s & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\\n',label,values);
end
fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n');
end
