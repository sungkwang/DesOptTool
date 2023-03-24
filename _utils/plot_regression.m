function R2_val = plot_regression(y_pred, y_true)
% plot regression of prediction results
hold on;

% standardization
% mean_val = mean(y_true);
% std_val = std(y_true);
% y_pred = (y_pred-mean_val)/std_val;
% y_true = (y_true-mean_val)/std_val;
y_pred = y_pred(:); y_true = y_true(:);

R2_val = get_rsquare(y_pred, y_true);
pid1 = plot(y_true, y_pred,'bx', 'MarkerSize',14);
p = polyfit(y_true,y_pred,1);
fit_line = polyval(p,[min(y_true(:)) max(y_true(:))]);
pid2 = plot([min(y_true(:)) max(y_true(:))], fit_line, '-b','LineWidth',2);
pid3 = plot([min(y_true(:)) max(y_true(:))], [min(y_true(:)) max(y_true(:))], '--k','LineWidth',2);
hold off; grid on; axis tight;
legend([pid1; pid2; pid3], {'data','fit','T=Y'},'Location','NW');
xlabel('Ground-Truth'); 
ylabel('Prediction');


