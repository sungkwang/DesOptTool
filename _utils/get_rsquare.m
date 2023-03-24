function rsquared = get_rsquare(y_pred, y_true)
% function to calculate r square value
rsquared = 1 - sum((y_true - y_pred).^2)/sum((y_true - mean(y_true)).^2);