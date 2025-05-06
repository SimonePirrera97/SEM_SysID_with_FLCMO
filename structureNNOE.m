function [Nn, N_layers] = structureIORNN(str)
    % Takes a string specifying the number of neurons per each HIDDEN layer. 
    Nn = str2num(str);
    N_layers = length(Nn);
end

