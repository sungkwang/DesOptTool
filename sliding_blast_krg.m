function [error_test, error_all] = sliding_blast_krg(num_training_data, basis_fun, random_seed_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~nargin
    set(0,'DefaultAxesFontName', 'Times New Roman','DefaultAxesFontSize', 18)
    set(0,'DefaultTextFontName', 'Times New Roman','DefaultTextFontSize', 18)    
    % number of training data set 
    num_training_data = 50;
    rng('default')
    clc;
    close all;
    random_seed_number = 1234;
    basis_fun = @dace_regpoly2;
    verbose = true;
else
    verbose = false;
end
% KRG_CorrelationModel =  @dace_corrgauss;
KRG_CorrelationModel =  @dace_correxp;

rng(random_seed_number); % to fixed the random seed for shuffling the data
% rng(11); % to fixed the random seed for shuffling the data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% basic information about the problem
% myFN = @forrester;  % this could be any user-defined function
% designspace = [0;   % lower bound
%                1];  % upper bound

% ndv = length(designspace(1,:));

% create DOE
% npoints = 10;
% X = linspace(designspace(1), designspace(2), npoints)';
% Y = feval(myFN, X);
    
filename_data = 'Blast_CoM_Analytical_Inches';
% filename_data = 'Impact_CoM_Analytical_Inches';
% filename_data = 'designA';

option.input_list = {'PanelLength','SteelThickness','EglassThickness'}; % as specified in the Matlab data format or excel csv file.
if contains(filename_data, 'Blast')
    option.output_list = {'Mass', 'RotationAngle', 'TotalLength'}; % as specified in the Matlab data format or excel csv file.
    option.major_obj_id = 2; % major objective index of output list for plotting purpose
elseif contains(filename_data, 'Impact')
    option.output_list = {'Mass', 'Displacement'}; % as specified in the Matlab data format or excel csv file.
    option.major_obj_id = 1; % major objective index of output list for plotting purpose
end


%%% load training/test data set based on split set
% For computational efficiency in Matlab, convert data from row major to column major
% x_train, y_train: trainining data based on the split ratio
% x_test, y_test: test data based on the split ratio
% x_all, y_all: all available data
% option.include_bound_points = false; % to include boundary points in training set. May improve
    
[x_train, y_train, x_test, y_test, x_all, y_all] = load_training_data(filename_data, num_training_data, option);

% % % read data
% filename_data = '_data/sliding_blast_v3_2_CoM_NoMS.mat';
% load(filename_data);
% DesignA = table2array(sliding_blast)';
% % 
% % random selection for training and testing
% ind_rand = randperm(length(DesignA));
% ind_training = ind_rand(1:num_training_data);
% ind_testing = ind_rand(num_training_data+1:end);
% 
% % ind_training = 1:num_training_data;
% % ind_testing = num_training_data+1:length(DesignA);
% 
% x_train = DesignA(1:3,ind_training)';  %x(1): l x(2): b x(3): t
% y_train = DesignA(4:6,ind_training)';
% 
% x_test = DesignA(1:3,ind_testing)';
% y_test = DesignA(4:6,ind_testing)';
% 
% % for total error
% x_all = DesignA(1:3,:)';
% y_all = DesignA(4:6,:)';
% 
% create test points
% npointstest = 101;
% Xtest = linspace(designspace(1), designspace(2), npointstest)';
% Ytest = feval(myFN, Xtest);

% X = x_train;
% Y = y_train;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fit surrogate
%   FIT_Fn - Function handle of the fitting function (which is used to
%            optimize KRG_Theta). [@dace_fit | @srgtsXVFit].
%            Default: @dace_fit.
%
%   KRG_RegressionModel  - Function handle to a regression model. [
%                          function_handle | @dace_regpoly0 |
%                          @dace_regpoly1 | @dace_regpoly2]. Default:
%                          @dace_regpoly0.

srgtOPT  = srgtsKRGSetOptions(x_train, y_train,  @dace_fit, basis_fun, KRG_CorrelationModel);
srgtSRGT = srgtsKRGFit(srgtOPT);
% 
% Xtest = x_test;
% Ytest = y_test;

[y_pred, PredVar] = srgtsKRGPredictor(x_test, srgtSRGT);
y_pred = y_pred';
y_test = y_test';
error_test = get_map_error(y_pred, y_test);



% [y_pred_all, PredVar] = srgtsKRGPredictor(x_all, srgtSRGT);
% y_pred_all = y_pred_all';
% y_all = y_all';
% error_test = get_map_error(y_pred_all, y_all);

[y, PredVar] = srgtsKRGPredictor(x_all, srgtSRGT);
y = y';
y_all = y_all';
error_all = get_map_error(y, y_all);

% alternatively, one can use
% Yhat    = srgtsKRGEvaluate(Xtest, srgtSRGT);
% PredVar = srgtsKRGPredictionVariance(Xtest, srgtSRGT);

% CRITERIA = srgtsErrorAnalysis(srgtOPT, srgtSRGT, y_test, y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % plots
% figure(1); clf(1);
% plot(X, Y, 'ok', ...
%     Xtest, Ytest, '--k', ...
%     Xtest, Yhat, '-b', ...
%     Xtest, Yhat + 2*sqrt(PredVar), 'r', ...
%     Xtest, Yhat - 2*sqrt(PredVar), 'r');
if verbose
    %plotting
    figure(1); hold on; grid on;
    plot( y(1,:), y(2,:),  'ro', 'MarkerSize',10);    
    plot(y_all(1,:), y_all(2,:), 'k.', 'MarkerSize',10);

    % create a directory for trained model
    directory_name = 'results/designA/';
    if ~exist(directory_name, 'dir')
        mkdir(directory_name);
    end    

    hold off; 
    legend('Ground Truth','Prediction','Location','NE');
    xlabel('Mass');
%     ylabel('Rot. angle');
    ylabel('Displacement');
        title(['Average percent error: ' num2str(error_test,2)]);
%     title(['RMS percent error: ' num2str(error_val,2)]);
    print('-dpng',[directory_name 'validation_designA_kriging.png']);
    
    
    figure(2); hold on; grid on;
    plot(y_pred(1,:), y_pred(2,:), 'ro', 'MarkerSize',10);    
    plot(y_test(1,:), y_test(2,:), 'k.', 'MarkerSize',10);

    % create a directory for trained model
    directory_name = 'results/designA/';
    if ~exist(directory_name, 'dir')
        mkdir(directory_name);
    end    

    hold off; 
    legend('Ground Truth','Prediction','Location','NE');
    xlabel('Mass');
%     ylabel('Rot. angle');
    ylabel('Displacement');
        title(['Average percent error: ' num2str(error_test,2)]);
%     title(['RMS percent error: ' num2str(error_val,2)]);
    print('-dpng',[directory_name 'validation_designA_kriging_test.png']);    
end    

function mape_val = get_map_error(y_pred, y_sol)
y_pred = y_pred(:);
y_sol = y_sol(:);
y_diff = (y_pred-y_sol)./y_sol;
y_diff(isnan(y_diff)) = 0;
mape_val = mean(abs(y_diff))*100;

function rmspe = get_rms_percent_error(y_pred, y_sol)
y_pred = y_pred(:);
y_sol = y_sol(:);
rmspe = sqrt(mean((y_pred-y_sol).^2./y_sol.^2))*100;


function [x_train, y_train, x_test, y_test, x_all, y_all] = load_training_data(filename_data, num_training_data, option)
% x_train, y_train: trainining data based on the split ratio
% x_test, y_test: test data based on the split ratio
% x_all, y_all: all available data

% requires input/output list of variable names, split ratio

input_list = option.input_list;
output_list = option.output_list;

% load training/validation data
filepath_data = ['_data/' filename_data];
if exist([filepath_data '.csv'],'file')
    raw_table = readtable([filepath_data '.csv']);
elseif exist([filepath_data '.xlsx'],'file')
    raw_table = readtable([filepath_data '.xlsx']);
else
    error('Data must be in csv or xlsx file');
end
num_total_data = size(raw_table,1);

% For computational efficiency in Matlab, convert data from row major to column major

input_raw_data = zeros(length(input_list), num_total_data);
% error handling for input/output variable names
for i=1:length(input_list)
    if ~any(strcmp(input_list{i}, raw_table.Properties.VariableNames))
        head(raw_table)
        error([input_list{i} ' does not match any column names in the table.']);
    end
    input_raw_data(i,:) = raw_table.(input_list{i});
end
output_raw_data = zeros(length(output_list),num_total_data);
for i=1:length(output_list)
    if ~any(strcmp(output_list{i}, raw_table.Properties.VariableNames))
        head(raw_table)
        error([output_list{i} ' does not match any column names in the table.']);
    end
    output_raw_data(i,:) = raw_table.(output_list{i});
end

%%% force to include boundary points in training set
if isfield(option, 'include_bound_points') && option.include_bound_points
    % column major approach
    % sort the data by input condition values.
    [~, ind] = sortrows(input_raw_data',[1 2 3]);
    input_raw_data = input_raw_data(:,ind);
    
    min_cond = min(input_raw_data,[],2);
    max_cond = max(input_raw_data,[],2);
    bound = [min_cond, max_cond];
    all_comb = allcomb(bound(1,:), bound(2,:), bound(3,:))';
    ind = [];
    for i=1:size(all_comb,1)
        % find the boundary values close to the bounds within 1e-5
        boundary_match = abs(input_raw_data - all_comb(:,i)) < 1e-5;
        ind_found = find(sum(boundary_match,1)==length(input_list)); % if all column
        ind = [ind, ind_found];
    end
    ind_map = false(1, size(input_raw_data,2));
    ind_map(ind) = 1;
    
    x_train1 = input_raw_data(:,ind_map);
    y_train1 = output_raw_data(:,ind_map);
    input_raw_data = input_raw_data(:,~ind_map);
    output_raw_data = output_raw_data(:,~ind_map);
    num_training_data = num_training_data - length(ind);
    
    %         %%%% row major approach, which is slower than column major
    %         % sort the data by input condition values.
    %         [~, ind] = sortrows(input_raw_data,[1 2 3]);
    %         input_raw_data = input_raw_data(ind,:);
    %
    %         min_cond = min(input_raw_data,[],1);
    %         max_cond = max(input_raw_data,[],1);
    %         bound = [min_cond', max_cond'];
    %         all_comb = allcomb(bound(1,:), bound(2,:), bound(3,:));
    %         ind = [];
    %         for i=1:size(all_comb,1)
    %             % find the boundary values close to the bounds within 1e-5
    %             boundary_match = abs(input_raw_data - all_comb(i,:)) < 1e-5;
    %             ind_found = find(sum(boundary_match,2)==length(input_list)); % if all column
    %             ind = [ind, ind_found];
    %         end
    %         ind_map = false(1, size(input_raw_data,1));
    %         ind_map(ind) = 1;
    %
    %         x_train1 = input_raw_data(ind_map,:);
    %         y_train1 = output_raw_data(ind_map,:);
    %         input_raw_data = input_raw_data(~ind_map,:);
    %         output_raw_data = output_raw_data(~ind_map,:);
    %
    %         num_training_data = num_training_data - length(ind);
    %         %%%% row major approach, which is slower than column major
    
else
    x_train1 = [];
    y_train1 = [];
end

% random selection for training and testing
ind_rand = randperm(size(input_raw_data,2));
ind_training = ind_rand(1:num_training_data);
ind_testing = ind_rand(num_training_data+1:end);

x_train = input_raw_data(:,ind_training);
y_train = output_raw_data(:,ind_training);

x_train = [x_train1 x_train]';
y_train = [y_train1 y_train]';

x_test = input_raw_data(:,ind_testing)';
y_test = output_raw_data(:,ind_testing)';

% for total error
x_all = input_raw_data';
y_all = output_raw_data';
