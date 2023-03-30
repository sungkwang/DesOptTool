function trained_model = run_surrogate_KRG_part_design(filename_data, num_training_data, x_train, y_train, option)
% This function trains Kriging based on samples drawn from training data
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
    option.instance = 1234;% must be positive integer
    rng(option.instance); % to fixed the random seed
    
    %%% setting for neural network
    % number of training data set; remaining will be used for validation
    num_training_data = 100;    
    % number of hidden layers and number of nodes
    basis_fun = @dace_regpoly2;
    KRG_CorrelationModel =  @dace_correxp;
            
    option.load_trained_model = false; % true: load trained model if exist
        
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
    option.row_major = true; % orient the data in row major layout for Kriging method
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
model_name = [directory_name  'trained_model_KRG_' filename_data '_' num2str(option.instance) '.mat'];
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
    srgtOPT  = srgtsKRGSetOptions(x_train, y_train,  @dace_fit, basis_fun, KRG_CorrelationModel);
    trained_model = srgtsKRGFit(srgtOPT);
    save(model_name, 'trained_model');
    disp(['KRG model has been saved to ' model_name]);
end

%%% plotting
if isplot
    toc;
    
    % get predictions 
    y_pred_train = srgtsKRGPredictor(x_train, trained_model);
    y_pred_test = srgtsKRGPredictor(x_test, trained_model);
    y_pred_all = srgtsKRGPredictor(x_all, trained_model);
    % get percent error
    error_train = get_mape(y_pred_train, y_train);
    error_test = get_mape(y_pred_test, y_test);
    error_all = get_mape(y_pred_all, y_all);
    disp(['Average percent error: ' num2str(error_all,2)]);
    
    % create a directory for trained model
    directory_name = ['results/' filename_data '/'];
    if ~exist(directory_name, 'dir'), mkdir(directory_name); end

    % convert row-major vectors for visualization
    x_train = x_train';
    x_test = x_test';
    x_all = x_all';
    y_train = y_train';
    y_test = y_test';
    y_all = y_all';
    y_pred_train = y_pred_train';
    y_pred_test = y_pred_test';
    y_pred_all = y_pred_all';
    
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
            print('-dpng',[directory_name 'validation_' filename_data '_' add_str '.png']);
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

function [y_pred, YY, ZZ] = prediction_design_space_3D(designspace, trained_model, option)
% predict some points in the design space to provice contour map
x = linspace(designspace(1,1),designspace(2,1),1);
y_pred_test = linspace(designspace(1,2),designspace(2,2),21);
z = linspace(designspace(1,3),designspace(2,3),21);
[XX,YY,ZZ] = ndgrid(x, y_pred_test, z);
x_pred = [XX(:) YY(:) ZZ(:)];
y_pred = srgtsKRGPredictor(x_pred, trained_model);
y_pred = reshape(y_pred(:,option.major_obj_id),size(XX));
