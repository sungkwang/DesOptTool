function rmse_val = get_rmse(y_pred, y_true)
% function to calculate root mean square error
y_diff = y_pred - y_true;
rmse_val = sqrt(mean(y_diff(:).^2));