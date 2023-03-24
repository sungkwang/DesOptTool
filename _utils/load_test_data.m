

function [x_test, y_test] = load_test_data(filename_data, option)
% x_test, y_test: test data for validation

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

x_test = input_raw_data;
y_test = output_raw_data;

