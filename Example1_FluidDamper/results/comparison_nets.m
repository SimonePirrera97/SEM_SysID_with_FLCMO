% Simone Pirrera
% Apr 17, 2024 
% Comparison of results from LM, CMO, and BPTT
clear; close all; clc;
fit_mean = zeros(3,1);
fit_std = zeros(3,1);
time_mean = zeros(3,1);
time_std = zeros(3,1);

name = ["NNOE (CMO)"; "NNOE (LM)"; "NNARX (Adam)"];

load("cmo_results_6neur_order4.mat");

fit_mean(1) = mean(FIT_cmo); fit_std(1) = std(FIT_cmo);
time_mean(1) = mean(time_cmo); time_std(1) = std(time_cmo);

load("nnsyd_results_6neur_order4.mat");

fit_mean(2) = mean(FIT_nnsyd); fit_std(2) = std(FIT_nnsyd); 
time_mean(2) = mean(time_nnsyd); time_std(2) = std(time_nnsyd);

data_nnarx = import_nnarxfile("results_nnarx_order4_6neur.txt",2);
fit_mean(3) = mean(data_nnarx(:,3)); fit_std(3) = std(data_nnarx(:,3)); 
time_mean(3) = mean(data_nnarx(:,1)); time_std(3) = std(data_nnarx(:,1));

time = strcat(num2str(time_mean), repmat(" (",3,1), num2str(time_std), repmat(")",3,1));
BFR = strcat(num2str(fit_mean), repmat(" (",3,1), num2str(fit_std), repmat(')',3,1));
%MSE = [num2str(fit_mean), repmat(' (',3,1), num2str(fit_std), repmat(')',3,1)];

T = table(name, time, BFR);
% Convert to LaTeX format
Ttex = table2latex(name,time, BFR);
writeToFile('tabella_es01_paper.tex',Ttex);


function [] = writeToFile(filename, filetext)
%writeToFile opens a new file with write options and writes the text on it

fileID = fopen(filename,'w');
fwrite(fileID, filetext);
fclose(fileID);

end 