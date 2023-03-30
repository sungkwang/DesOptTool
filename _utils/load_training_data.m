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

% error handling for input/output variable names
input_raw_data = [];
for i=1:length(input_list)
    if ~any(strcmp(input_list{i}, raw_table.Properties.VariableNames))
        head(raw_table)
        error([input_list{i} ' does not match any column names in the table.']);
    end
    input_raw_data.(input_list{i})  = raw_table.(input_list{i});
end
output_raw_data = [];
for i=1:length(output_list)
    if ~any(strcmp(output_list{i}, raw_table.Properties.VariableNames))
        head(raw_table)
        error([output_list{i} ' does not match any column names in the table.']);
    end
    output_raw_data.(output_list{i}) = raw_table.(output_list{i});
end

% convert struct to array
input_raw_data = struct2array(input_raw_data);
output_raw_data = struct2array(output_raw_data);

% remove rows that contain nan values
ind_to_exclude = any(isnan(input_raw_data),2) | any(isnan(output_raw_data),2);
num_to_exclude = length(find(ind_to_exclude));
disp(['Not-a-number/empty data: ' num2str(num_to_exclude) ' are removed from training database. Remaining: ' num2str(num_total_data-num_to_exclude)]);
input_raw_data(ind_to_exclude,:)  = [];
output_raw_data(ind_to_exclude,:)  = [];

% remove rows that contain redundant data
ind_to_exclude = any(isnan(input_raw_data),2) | any(isnan(output_raw_data),2);
num_to_exclude = length(find(ind_to_exclude));
disp(['Not-a-number/empty data: ' num2str(num_to_exclude) ' are removed from training database. Remaining: ' num2str(num_total_data-num_to_exclude)]);
input_raw_data(ind_to_exclude,:)  = [];
output_raw_data(ind_to_exclude,:)  = [];

[u,I,J] = unique(input_raw_data, 'rows', 'first');

ixDupRows = setdiff(1:size(input_raw_data,1), I)
dupRowValues = input_raw_data(ixDupRows,:)

% min_vals = min(input_raw_data);
% max_vals = max(input_raw_data);

input_raw_data = input_raw_data(I,:)';
output_raw_data = output_raw_data(I,:)';

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

x_train = [x_train1 x_train];
y_train = [y_train1 y_train];

x_test = input_raw_data(:,ind_testing);
y_test = output_raw_data(:,ind_testing);

% for total error
x_all = input_raw_data;
y_all = output_raw_data;

if isfield(option,'row_major') && option.row_major
    x_train = x_train';
    x_test = x_test';
    x_all = x_all';
    y_train = y_train';
    y_test = y_test';
    y_all = y_all';    
end
