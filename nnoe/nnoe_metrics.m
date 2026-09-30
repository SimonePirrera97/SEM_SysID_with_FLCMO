function [rmse,bfr] = nnoe_metrics(reference,prediction)
%NNOE_METRICS Per-output RMSE and best-fit rate in percent.
error=prediction-reference;
rmse=sqrt(mean(error.^2,1));
bfr=zeros(1,size(reference,2));
for outputIndex=1:size(reference,2)
    centered=reference(:,outputIndex)-mean(reference(:,outputIndex));
    bfr(outputIndex)=100*(1-norm(error(:,outputIndex))/norm(centered));
end
end
