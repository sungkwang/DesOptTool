function trained_model = run_surrogate_ANN_part_design(filename_data, num_training_data, ...
    hidden_layer, x_train, y_train, option)
% This function trains a neural network based on samples drawn from training data
%
% Authors/Contacts:
%    Sungkwang Mun <sungkwan@cavs.msstate.edu>
%
addpath('_utils');

if ~nargin
    set(0,'DefaultAxesFontName', 'Times New Roman','DefaultAxesFontSize', 18)
    set(0,'DefaultTextFontName', 'Times New Roman','DefaultTextFontSize', 18)
    close all;
    
    % to contrrol random number generator by initializing with a positive integer value
    option.instance = 1234; % must be positive integer
    rng(option.instance); % to fixed the random seed
    
    %%% setting for neural network
    % number of training data set; remaining will be used for validation
    num_training_data = 100;
    
    % number of hidden layers and number of nodes
%     num_nodes = 10;
%     num_layers = 5; % hidden layer depth
%     hidden_layer = ones(1, num_layers)*num_nodes;
    % or directly define layer structure
    hidden_layer = [20 20];
    % or directly define layer structure per object
%     hidden_layer{1} = [8 3]; 
%     hidden_layer{2} = [10 3]; 
%     hidden_layer{3} = [3 1]; 
    
    option.load_trained_model = false; % true: load trained model if exist
        
    % split ratio used for training. igrore this for now. change later
    option.trainRatio = 0.7; % default
    option.valRatio = 0;   % should be zero for Bayesian regularization
    option.testRatio = 0.3;
    %%% setting for neural network
    
    %%% setting for performance improvements
    % ensemble learning options
    option.is_ensemble = false ; % true: load trained ensemble model. igrore this for now
    option.instance_list = [1 2 3]; % need to include igrore this for now
    option.is_single_net = false; % true: single network, false: multi-network (NN per each output) igrore this for now
    %%% setting for performance improvements
    
    %%% specification about problem
    filename_data = 'designA';
    
    option.input_list = {'Length','Width','Thickness'}; % as specified in the Matlab data format or excel csv file.
    option.output_list = {'Mass', 'Stress', 'BucklingLoad'}; % as specified in the Matlab data format or excel csv file.
    option.major_obj_id = 1; % major objective index of output list for plotting purpose
    
    %%% specification about problem
    
    %%% load training/test data set based on split set
    % For computational efficiency in Matlab, convert data from row major to column major
    % x_train, y_train: trainining data based on the split ratio
    % x_test, y_test: test data based on the split ratio
    % x_all, y_all: all available data
    option.include_bound_points = false; % to include boundary points in training set. May improve
    [x_train, y_train, x_test, y_test, x_all, y_all] = load_training_data(filename_data, num_training_data, option);
    
    tic;
    isplot = true;
else
    % to control the random number generator for split ratio and initialzation of weights
    rng(option.instance); % to fixed the random seed
    isplot = false;
end

% create a directory for trained model
directory_name = ['./trained_model/' filename_data '/'];
if ~exist(directory_name, 'dir'), mkdir(directory_name); end

% standardization
x_mean = mean(x_train, 2);
x_std = std(x_train, 0, 2);
x_train_std = x_train;
x_train_std = (x_train_std-x_mean)./x_std;

y_mean = mean(y_train, 2);
y_std = std(y_train, 0, 2);
y_train_std = y_train;
y_train_std = (y_train_std-y_mean)./y_std;

train_function = 'trainbr'; % bayesian regularization. 
% train_function = 'trainlm'; % BFGS

option.epochs = 5000;
% option.min_grad = 1e-6;
% option.showCommandLine = true;

% load the trained network if needed
if iscell(hidden_layer)
    hidden_layer_str = [];
    for i=1:length(hidden_layer)
        hidden_layer_str = [hidden_layer_str strrep(num2str(hidden_layer{i}),'  ' ,'_') '-'];
    end
else
    hidden_layer_str = strrep(num2str(hidden_layer),'  ' ,'_');
end
train_str = ['_hl_' hidden_layer_str '_nd_' num2str(num_training_data)  ...
    '_tr_' num2str(option.trainRatio) '_' num2str(option.valRatio) '_' num2str(option.testRatio) ];
if option.is_single_net,  train_str = [train_str  '_single']; end

if option.is_ensemble
    % ensemble learning requires instance number list
    if ~isfield(option,'instance_list')
        error('Ensemble Learning: requires list of instance numbers. Run multiple times with diferent instance number');
    else
        disp(['Ensemble Learning: ' num2str(length(option.instance_list)) ' NNs will be loaded.']);
    end
    
    % load trained networks and each network's mean and standard deviation
    num_trials = length(option.instance_list);
    trained_model = cell(1,num_trials);
%     trained_model.nets_all = cell(length(option.output_list),num_trials); trained_model.y_means = cell(1,num_trials); trained_model.y_stds = cell(1,num_trials);
    for i = 1:num_trials
        instance_temp = option.instance_list(i);
        % draw samples from randomly shuffled and divided training/testing data
        disp(['Ensemble Learning: Loading NN with random seed ' num2str(instance_temp)]);
        model_name = [directory_name  'trained_model_ANN_' filename_data '_' train_str '_i_' num2str(instance_temp) '.mat'];
        [model, option] = load_model_ensemble(model_name, option);
        trained_model{i}.nets = model.nets;
        trained_model{i}.y_mean = model.y_mean;
        trained_model{i}.y_std = model.y_std;
    end
    % save combined networks
    model_name = [directory_name  'trained_model_ANN_' filename_data '_' train_str '_combined.mat'];
    is_single_net = option.is_single_net;
    save(model_name, 'trained_model', 'is_single_net');    
    disp(['Combined_model has been saved to ' model_name]);
else
    model_name = [directory_name  'trained_model_ANN_' filename_data '_' train_str '_i_' num2str(option.instance) '.mat'];
    if option.load_trained_model
        % load pre-trained model
        if exist(model_name,'file')
            load(model_name, 'trained_model', 'is_single_net');
            option.is_single_net = is_single_net;
            disp(['Pretrained model loaded. Model location: ' model_name]);
        else
            error('No pre-trained model exist. Rerun with load_trained_model=false')
        end
    else % if there is no pre-trained model, then train from scratch
        num_outputs = length(option.output_list);
        nets = build_nets(hidden_layer, train_function, num_outputs, option);
        disp([num2str(length(nets)) ' networks will be generated...']);
        % train the network
        nets = train_nets(nets, x_train_std, y_train_std, option);
        % save trained network
        is_single_net = option.is_single_net;
        trained_model.nets = nets;
        trained_model.x_std = x_std;
        trained_model.x_mean = x_mean;     
        trained_model.y_std = y_std;
        trained_model.y_mean = y_mean;     
        
        save(model_name, 'trained_model', 'is_single_net');
        disp(['NN model has been saved to ' model_name]);
    end
end


%%% plotting
if isplot
    toc;
    
    % plot that shows how the data is split into training/test
    fig = figure(1); fig.Position=[100 100 800 600]; hold on;
    plot3(x_train(1,:),x_train(2,:),x_train(3,:),'ko','MarkerSize',8, 'MarkerFaceColor', 'r');
    plot3(x_test(1,:),x_test(2,:),x_test(3,:),'ks', 'MarkerSize',20, 'MarkerFaceColor', 'b');
    hold off; grid on;
    xlabel(option.input_list{1});
    ylabel(option.input_list{2});
    zlabel(option.input_list{3});
    legend('training','test');
    axis tight;
    view(3);    
    
    % get predictions 
    if option.is_ensemble
        y_pred_train = predict_nets_ensemble(x_train, trained_model, option);
        y_pred_test = predict_nets_ensemble(x_test, trained_model, option);
        y_pred_all = predict_nets_ensemble(x_all, trained_model, option);
    else
        y_pred_train = predict_nets(x_train, trained_model, option);
        y_pred_test = predict_nets(x_test, trained_model, option);
        y_pred_all = predict_nets(x_all, trained_model, option);
    end
    % get percent error
    error_train = get_mape(y_pred_train, y_train);
    error_test = get_mape(y_pred_test, y_test);
    error_all = get_mape(y_pred_all, y_all);
    disp(['Average percent error: ' num2str(error_all,2)]);
    
    % create a directory for trained model
    directory_name = ['results/' filename_data '/'];
    if ~exist(directory_name, 'dir'), mkdir(directory_name); end
    
    %%%% plotting results
    fig = figure(2); fig.Position=[100 100 1600 1400];
    tiledlayout(2,2); % two subplots in one figure

    % prediction comparison using train data only
    nexttile;
    R2_val = plot_regression(y_pred_train, y_train);
    title(['Training data, R^2= ' num2str(R2_val,2) ', MAPE= ' num2str(error_train,2)]);
    
    % prediction comparison using test data only
    nexttile;
    R2_val = plot_regression(y_pred_test, y_test);
    title(['Test data, R^2= ' num2str(R2_val,2) ', MAPE= ' num2str(error_test,2)]);

    % prediction comparison using all data
    nexttile;
    R2_val = plot_regression(y_pred_all,y_all);
    title(['All data, R^2= ' num2str(R2_val,2) ', MAPE= ' num2str(error_all,2)]);
    
    if length(option.output_list)>2
        non_major_obj_id = find(option.major_obj_id ~= 1:length(option.output_list));
        for i=1:length(non_major_obj_id)
            non_target_id = non_major_obj_id(i);
            add_str = [option.output_list{non_target_id} '_' option.output_list{option.major_obj_id} ];

            fig = figure(i+2); fig.Position=[100 100 1600 600];
            tiledlayout(1,2); % two subplots in one figure

            % prediction comparison using test data only
            nexttile;
            hold on;
            plot(y_test(non_target_id,:), y_test(option.major_obj_id,:), 'ro', 'MarkerSize',10);
            plot(y_pred_test(non_target_id,:), y_pred_test(option.major_obj_id,:), 'k.', 'MarkerSize',10);
            %     rms_error = sqrt(mean((y_pred_test(:)-y_test(:)).^2./y_test(:).^2))*100;
            hold off; grid on;
            legend('Ground-Truth','ML prediction','Location','NE');
            xlabel(option.output_list{non_target_id});
            ylabel(option.output_list{option.major_obj_id});
            title(['Test data, Average percent error: ' num2str(error_test,2)]);

            % prediction comparison using all data
            nexttile;
            hold on;
            plot(y_all(non_target_id,:), y_all(option.major_obj_id,:), 'ro', 'MarkerSize',10);
            plot(y_pred_all(non_target_id,:), y_pred_all(option.major_obj_id,:), 'k.', 'MarkerSize',10);
            %     rms_error = sqrt(mean((y_pred_test(:)-y_test(:)).^2./y_test(:).^2))*100;
            hold off; grid on;
            legend('Ground-Truth','ML prediction','Location','NE');
            xlabel(option.output_list{non_target_id});
            ylabel(option.output_list{option.major_obj_id});
            title(['All data, Average percent error: ' num2str(error_all,2)]);

            % save to graphic file
            print('-dpng',[directory_name 'validation_' filename_data train_str '_' add_str '.png']);
        end
    end
    
    % contour plot if three design variables are used
    if length(option.input_list)==3
        % contour prediction for min of first variable
        min_first_var = min(x_all(1,:)); 
        designspace = [min_first_var min(x_all(2,:)) min(x_all(3,:));
                       min_first_var max(x_all(2,:)) max(x_all(3,:))];
        y_pred_min = prediction_design_space_3D(designspace, trained_model, option);

        % contour prediction for median of first variable
        med_first_var = median(x_all(1,:)); 
        designspace = [med_first_var min(x_all(2,:)) min(x_all(3,:));
                       med_first_var max(x_all(2,:)) max(x_all(3,:))];
        y_pred_med = prediction_design_space_3D(designspace, trained_model, option);

        % contour prediction for max of first variable
        max_first_var = max(x_all(1,:)); 
        designspace = [max_first_var min(x_all(2,:)) min(x_all(3,:));
                       max_first_var max(x_all(2,:)) max(x_all(3,:))];
        [y_pred_max, YY, ZZ] = prediction_design_space_3D(designspace, trained_model, option);;

        % plot predicted contours with ground truth data
        fig = figure(10); fig.Position=[100 100 1600 1200]; 
        tiledlayout(2,2); % two subplots in one figure

        nexttile; hold on; 
        contour3(squeeze(YY(1,:,:)),squeeze(ZZ(1,:,:)),squeeze(y_pred_min(1,:,:)),50, 'LineWidth',2);
        %     surf(squeeze(YY(1,:,:)),squeeze(ZZ(1,:,:)),squeeze(y_pred_min(1,:,:)),squeeze(y_pred_min(1,:,:)));
        ind = x_all==min_first_var; ind = ind(1,:); % select only data that matches with min_first_var
        x = x_all(:,ind);
        y_pred_test = y_all(option.major_obj_id,ind);
        ypred = y_pred_all(option.major_obj_id,ind);
        plot3(x(2,:), x(3,:), ypred, 'k.','MarkerSize',10);
        plot3(x(2,:), x(3,:), y_pred_test, 'ro','MarkerSize',10);
        xlabel(option.input_list{2}); ylabel(option.input_list{3});  zlabel(option.output_list{option.major_obj_id}); 
        title(['Min value for ' option.input_list{1} ': ' num2str(min_first_var)]);
        legend('Predicted countour', 'Predicted point', 'Ground-truth');
        hold off; grid on;
        view(23,20);
    %     view(125,30); 

        nexttile; hold on;  
        contour3(squeeze(YY(1,:,:)),squeeze(ZZ(1,:,:)),squeeze(y_pred_med(1,:,:)),50, 'LineWidth',2);
        ind = x_all==med_first_var; ind = ind(1,:); % select only data that matches with med_first_var
        x = x_all(:,ind);
        y_pred_test = y_all(option.major_obj_id,ind);
        ypred = y_pred_all(option.major_obj_id,ind);
        plot3(x(2,:), x(3,:), ypred, 'k.','MarkerSize',10);
        plot3(x(2,:), x(3,:), y_pred_test, 'ro','MarkerSize',10);
        xlabel(option.input_list{2}); ylabel(option.input_list{3});  zlabel(option.output_list{option.major_obj_id}); 
        title(['Median value for ' option.input_list{1} ': ' num2str(med_first_var)]);
        legend('Predicted countour', 'Predicted point', 'Ground-truth');
        hold off; grid on;
        view(23,20);
    %     view(125,30); 

        nexttile; hold on;
        ind = x_all==max_first_var; ind = ind(1,:); % select only data that matches with max_first_var
        x = x_all(:,ind);
        y_pred_test = y_all(option.major_obj_id,ind);
        ypred = y_pred_all(option.major_obj_id,ind);
        if ~isempty(ind)
            plot3(x(2,:), x(3,:), ypred, 'k.','MarkerSize',10);
            plot3(x(2,:), x(3,:), y_pred_test, 'ro','MarkerSize',10);
        end
        contour3(squeeze(YY(1,:,:)),squeeze(ZZ(1,:,:)),squeeze(y_pred_max(1,:,:)),50, 'LineWidth',2);
        xlabel(option.input_list{2}); ylabel(option.input_list{3});  zlabel(option.output_list{option.major_obj_id}); 
        title(['Max value for ' option.input_list{1} ': ' num2str(max_first_var)]);
        if ~isempty(ind)
            legend('Predicted countour', 'Predicted point', 'Ground-truth');
        end
        hold off; grid on;
        view(23,20);
    %     view(125,30); 

        nexttile; hold on; % put all together
        contour3(squeeze(YY(1,:,:)),squeeze(ZZ(1,:,:)),squeeze(y_pred_min(1,:,:)),50,'-r');
        contour3(squeeze(YY(1,:,:)),squeeze(ZZ(1,:,:)),squeeze(y_pred_med(1,:,:)),50, '--g');
        contour3(squeeze(YY(1,:,:)),squeeze(ZZ(1,:,:)),squeeze(y_pred_max(1,:,:)),50, ':b');
        hold off; grid on; 
        view(23,20);
        xlabel(option.input_list{2}); ylabel(option.input_list{3});  zlabel(option.output_list{option.major_obj_id}); 
        legend(num2str(min_first_var),num2str(med_first_var),num2str(max_first_var));
        title('Combined');

        print('-dpng',[directory_name 'countour_' filename_data '_' num2str(num_training_data) '.png']);
    end    
end

function [trained_model, option] = load_model_ensemble(model_name, option)
if exist(model_name,'file')
    load(model_name, 'trained_model', 'is_single_net');
    option.is_single_net = is_single_net;
    disp(['Ensemble Learning: Pretrained model loaded, single net = ' num2str(is_single_net)]);
else
    error(['Ensemble Learning: no file exist. Train with no ensemble option with instance number= ' num2str(instance_temp)])
end

function nets = build_nets(hidden_layer, train_function, num_outputs,option)
% build either a single networks for all outputs or multiple networks,i.e., network per each output.
% option.is_single_net: true - single netowrk, false - multiple networks

nets = cell(1, num_outputs);
for i=1:num_outputs
    if iscell(hidden_layer)
        nets{i} = feedforwardnet(hidden_layer{i}, train_function); % bayesian regularization
    else
        nets{i} = feedforwardnet(hidden_layer, train_function); % bayesian regularization
    end
%     nets{i}.layers{1:end-1}.transferFcn = 'tansig'; %'tansig'
%     nets{i}.layers{end}.transferFcn = 'tansig';
    
%     net.performParam.regularization = 1e-6;
    % net.trainParam.max_fail = 1000; % this enables the use of validation set but slow and poor accuracy
    
%     nets{i}.trainParam.goal = 1e-5;
%     nets{i}.trainParam.max_fail = 100;
    
%     nets{i}.trainParam.mu = 0.005;
    if isfield(option, 'trainRatio') && isfield(option, 'valRatio') && isfield(option, 'testRatio')
        nets{i}.divideParam.trainRatio = option.trainRatio;
        nets{i}.divideParam.valRatio = option.valRatio;
        nets{i}.divideParam.testRatio = option.testRatio;
    end
    
    if isfield(option, 'epochs')
        nets{i}.trainParam.epochs = option.epochs;
    end
    
    if isfield(option, 'showCommandLine')
        nets{i}.trainParam.showCommandLine = option.showCommandLine;
    end
    
    if isfield(option, 'min_grad')
        nets{i}.trainParam.min_grad = option.min_grad;
    end
    
    nets{i}.trainParam.showWindow =true;
%     if isfield(option, 'no_gui')
%         nets{i}.trainParam.showWindow = false;
%     end
    if option.is_single_net
        break;
    end
end

function nets = train_nets(nets, x, y_pred_test, option)
% train NN for single network for all outputs or NN per each output
if option.is_single_net
    nets{1} = train(nets{1}, x, y_pred_test);
else
    for i=1:size(y_pred_test,1)
        disp(['Training NN for output ' num2str(i) ' ' option.output_list{i} ' with random seed number ' num2str(option.instance) ' started...']);
        [nets{i}, tr] = train(nets{i}, x, y_pred_test(i,:));
        disp('Training NN done...');
        
        % Once the network has been trained, we can obtain the Mean Squared Error
        % for the best epoch (time when the training has stopped in order to avoid
        % overfitting the network).
        mse_train = tr.perf(tr.best_epoch + 1); % There is epoch 0, but arrays in 
                                                % MATLAB start in 1.
        mse_val = tr.vperf(tr.best_epoch + 1);
        mse_test = tr.tperf(tr.best_epoch + 1);        
    end
end

function y_pred_test = predict_nets(x, trained_model, option)
% predict using single network for all outputs or multiple NN per each output
nets = trained_model.nets; 
x_mean = trained_model.x_mean; x_std = trained_model.x_std; 
y_mean = trained_model.y_mean; y_std = trained_model.y_std; 
x = (x-x_mean)./x_std;
if option.is_single_net
    y_pred_test = nets{1}(x);
    % inverse standardization
    y_pred_test = (y_pred_test.*y_std) + y_mean;
else
    y_pred_test = zeros(length(nets), size(x,2));
    for i=1:length(nets)
        y_pred_test(i,:) = nets{i}(x);
    end
    y_pred_test = (y_pred_test.*y_std) + y_mean;
end

function [y_combine, y_all]= predict_nets_ensemble(x, trained_model, option)
% predict using combination of multiple NN that are obtained from different split ratio/initialization.
% nets_all = trained_model.nets_all; y_means = trained_model.y_means; y_stds = trained_model.y_stds;
y_all = cell(1, length(trained_model));
y_combine = zeros(length(trained_model{1}), size(x,2));
for k = 1:length(trained_model)
    y_pred_test = predict_nets(x, trained_model{k}, option);
    y_all{k} = y_pred_test;
    y_combine = y_combine + y_pred_test;
end
y_combine = y_combine/length(trained_model);

function [y_pred, YY, ZZ] = prediction_design_space_3D(designspace, trained_model, option)
% predict some points in the design space to provice contour map
x = linspace(designspace(1,1),designspace(2,1),1);
y_pred_test = linspace(designspace(1,2),designspace(2,2),21);
z = linspace(designspace(1,3),designspace(2,3),21);
[XX,YY,ZZ] = ndgrid(x, y_pred_test, z);
x_pred = [XX(:) YY(:) ZZ(:)]';
if option.is_ensemble
    y_pred = predict_nets_ensemble(x_pred, trained_model, option);
else
    y_pred = predict_nets(x_pred, trained_model, option);
end
y_pred = reshape(y_pred(option.major_obj_id,:),size(XX));
