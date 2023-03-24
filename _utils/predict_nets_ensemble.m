function [y_combine, y_all]= predict_nets_ensemble(x, trained_model, option)
% predict using combination of multiple NN that are obtained from different split ratio/initialization.
% nets_all = trained_model.nets_all; y_means = trained_model.y_means; y_stds = trained_model.y_stds;
y_all = cell(1, length(trained_model));
num_models = length(trained_model);
num_obj = length(trained_model{1}.nets);
y_combine = zeros(num_obj, size(x,2));
for k = 1:length(trained_model)
    y = predict_nets(x, trained_model{k}, option);
    y_all{k} = y;
    y_combine = y_combine + y;
end
y_combine = y_combine/num_models;
