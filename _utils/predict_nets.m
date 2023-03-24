function y = predict_nets(x, trained_model, option)
% predict using single network for all outputs or multiple NN per each output
nets = trained_model.nets; 
x_mean = trained_model.x_mean; x_std = trained_model.x_std; 
y_mean = trained_model.y_mean; y_std = trained_model.y_std; 
x = (x-x_mean)./x_std;
if option.is_single_net
    y = nets{1}(x);
    % inverse standardization
    y = (y.*y_std) + y_mean;
else
    y = zeros(length(nets), size(x,2));
    for i=1:length(nets)
        y(i,:) = nets{i}(x);
    end
    y = (y.*y_std) + y_mean;
end