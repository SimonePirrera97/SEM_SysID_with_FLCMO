clear; close all; clc;

load("..\data\dataset_test_03.mat");
load('NNOE\res03_2_5.mat');
load('LSTM\lstmtest2_2.mat');
load('GRU\grutest1_5.mat');

Ntest = size(y_test,1);

cd ..\ml_mimo_nnoe_p
y_nnoe = simulateMultiLayerNNOE(Net, u_test, y_test(1:3,:));
cd ..\'results dataset03'


for idx = 1:2
figure, 
    plot(1:Ntest, y_test(:,idx)), hold on;
    plot(1:Ntest, y_nnoe(:,idx)), 
    plot(1:Ntest, lstmtest22(:,idx)), 
    plot(1:Ntest, grutest15(:,idx)); grid on;

legend('True', 'NNOE(2,3)', 'LSTM(2,2)', 'GRU(1,5)',...
    'FontSize', 24,'Interpreter', 'latex');

xlabel('Time, $t$','FontSize', 24,'Interpreter', 'latex');
ylabel('$\hat y_t$','FontSize', 24,'Interpreter', 'latex');
end
