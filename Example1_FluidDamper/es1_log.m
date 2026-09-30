function es1_log(cfg,operation,result)
%ES1_LOG Report one completed operation to the terminal and results.txt.
line=sprintf('[%s] %s | result: %s',datestr(now,'yyyy-mm-dd HH:MM:SS'),operation,result);
fprintf('%s\n',line);
fid=fopen(cfg.logFile,'a');
assert(fid~=-1,'Cannot open log file: %s',cfg.logFile);
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',line);
end
