function mape_val = get_mape(y_pred, y_true)
y_pred = y_pred(:);
y_true = y_true(:);
y_diff = (y_pred-y_true)./y_true;
y_diff(isnan(y_diff)) = 0;
mape_val = mean(abs(y_diff))*100;
