function [rmse,fit] = es1_metrics(y,yhat,order)
% Ignore the supplied initial-condition samples in every simulation score.
y=y(order+1:end,:); yhat=yhat(order+1:end,:);
rmse=sqrt(mean((yhat-y).^2,'all'));
fit=100*(1-norm(yhat-y)/norm(y-mean(y,1)));
end
