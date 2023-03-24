function [hlink1, hlink2] = run_optimization_ternary_part_design
% This function provides ternary analysis plot based on the
% final Pareto frontiers of the GA results to narrow down the optimal design
% candidates with respect to level of importance/compromise of each objective.
% 
% Authors/Contacts:
%    Sungkwang Mun <sungkwan@cavs.msstate.edu>
%
set(0,'DefaultAxesFontName', 'Times New Roman','DefaultAxesFontSize', 18)
set(0,'DefaultTextFontName', 'Times New Roman','DefaultTextFontSize', 18)
close all; clear;
addpath(genpath('_utils'));

%% parameters for ternary analysis
% copy the GA result location from the ANN training script to the model name
% variable below
result_name = 'results/GA/ANN_nPop_500_nIter_100.mat';

ternary_obj_ids = [1 2 3]; % three objectives can be specified if the number of objectives are more than three.
% decimal places for the values on the contour lines
decimal_places_contour_lines = [4 4 4]; % change the numbers to have different decimal place, any integer grater than 0.
acceptable_percents = [38 40 20]; % define acceptable level in percent for all objectives, range 0~100%
% if you see some weid plots, turn off is_arrow option
is_arrow = true; % true: add arrows on each contour line, false: no arrows

num_base_points = 21; % increasing this will increase the resolution of contour plot

%% load results file and assign to some local vairables.
load(result_name, 'option','result_Optim');
method = option.method;
var_str = option.nameVar;
obj_str = option.nameObj;
num_population = option.popsize;
num_iteration = option.maxGen;

% minimize/maximize indicator for visualization purpose on ternary diagrm
% it will flip the sign of GA results whose objectives are to be maximized
is_maximize = option.is_maximize;
is_maximize = logical(is_maximize);
option.is_arrow = true; % true: add arrows on each contour line, false: no arrows

% get final pareto front responses/objectives and variables from GA
final_gen = result_Optim.pops(option.maxGen,:);
responses = reshape([final_gen.obj], [option.numObj option.popsize])';
variables = reshape([final_gen.var], [option.numVar option.popsize])';

disp(['List of Variables: ' var_str]);
disp(['List of Objectives: ' obj_str]);

num_objectives = length(obj_str);

directory_name = 'results/ternary';
if ~exist(directory_name, 'dir'), mkdir(directory_name); end
filename = [method '_nPop_' num2str(num_population) '_nIter_' num2str(num_iteration)];

weight_str = cell(1,3);
for i=1:3
    weight_str{i} = ['Weight on ' obj_str{i}];
end
% if you see some weid plots, turn off is_arrow option
option.is_arrow = is_arrow; % true: add arrows on each contour line, false: no arrows

%% prepare weight matrix for ternay analysis
w = linspace(0,1,num_base_points);
w = sort([w 0.333]); % add center point (0.333) and sort

% construct full factorial design and select only the weight sum being one.
ff = fullfact([length(w) length(w) length(w)]);
ws = sortrows(w(ff)); 
inds= abs(sum(ws,2)-1)<1e-3; % find only sum of all weights are one.
ws = ws(inds,:);

% % %%% original weight matrix
% ws = [
%     0	0	1
%     0	1	0
%     1	0	0
%     0.5	0.5	0
%     0	0.5	0.5
%     0.5	0	0.5
%     0.33	0.33	0.34
%     0.2	0	0.8
%     0.2	0.8	0
%     0.8	0.2	0
%     0.8	0	0.2
%     0.4	0	0.6
%     0.4	0.6	0
%     0.6	0	0.4
%     0.6	0.4	0
% ];

% zero padding to rule one objective out from 4 objectives. Change this where the zeros are padded.
ws_padded = zeros(size(ws,1), num_objectives);
ws_padded(:,ternary_obj_ids) = ws; % plug in the corresponding weights while setting zero for objective ignored
ws_str = cell(1, num_objectives);
ws_str(:) = {'w0'};
ws_str(ternary_obj_ids) = {'w1','w2','w3'};

% get minimum distance when it accounts for weights for each response
opt_ind = zeros(1, size(ws_padded,1));
for i=1:size(ws_padded,1)
    w = ws_padded(i,:);
    distances = get_weighted_distance(responses, w);
    % get the index of the minimum distanced one from Pareto Front
    [~, opt_ind(i)] = min(distances);
end


%% prepare data for ternary plot
d1 = responses(opt_ind, ternary_obj_ids(1));
d2 = responses(opt_ind, ternary_obj_ids(2));
d3 = responses(opt_ind, ternary_obj_ids(3));
% to choose contour line in term of acceptable percentage level within min/max of response
zval1 = min(d1) + (max(d1) - min(d1))*acceptable_percents(ternary_obj_ids(1))/100;
zval1 = round(zval1, decimal_places_contour_lines(ternary_obj_ids(1)));
zval2 = min(d2) + (max(d2) - min(d2))*acceptable_percents(ternary_obj_ids(2))/100;
zval2 = round(zval2, decimal_places_contour_lines(ternary_obj_ids(2)));
zval3 = min(d3) + (max(d3) - min(d3))*acceptable_percents(ternary_obj_ids(3))/100;
zval3 = round(zval3, decimal_places_contour_lines(ternary_obj_ids(3)));
option.zvals = [zval1 zval2 zval3]; % the threshold level on each objective for ternary analysis

% for points within intersection regions. See tercontour function
variables_opt = variables(opt_ind,:);
responses_opt = responses(opt_ind,:);
ind_inter = responses_opt(:,ternary_obj_ids(1))<=zval1 ...
          & responses_opt(:,ternary_obj_ids(2))<=zval2 ...
          & responses_opt(:,ternary_obj_ids(3))<=zval3;
option.ind_inter = ind_inter;

% flip the sign for any variable for maximization (minimization of negative)
responses(:,is_maximize) = -responses(:,is_maximize);
option.zvals(is_maximize) = -option.zvals(is_maximize);
if is_maximize(1), d1 = -(d1); end
if is_maximize(2), d2 = -(d2); end
if is_maximize(3), d3 = -(d3); end
responses_opt(is_maximize) = -responses_opt(is_maximize);

%% print all results on display and save them in excel file
result_all = [ws_padded responses(opt_ind,:) variables(opt_ind,:)];
str_all = {ws_str{:}, obj_str{:} var_str{:}};
result_table = array2table(result_all);
result_table.Properties.VariableNames =  str_all;

disp('*** Ternary plot analysis result ')
disp(result_table);
filename_ternary = [directory_name '/' 'ternary_analysis_' method '.xls'];
writetable(result_table,filename_ternary, 'WriteMode', 'replacefile')
disp(['*** Ternary plot analysis results also save to ''' filename_ternary '''']);

%% print select results within intersection on display and save them in excel file
result_intersection = [ws_padded(ind_inter,:) responses_opt(ind_inter,:) variables_opt(ind_inter,:)];
result_table = array2table(result_intersection);
result_table.Properties.VariableNames =  str_all;

disp('*** Ternary plot analysis result ')
disp(result_table);
filename_ternary = [directory_name '/'  'ternary_analysis_' method '_intersection.xls'];
writetable(result_table,filename_ternary, 'WriteMode', 'replacefile')
disp(['*** Ternary plot analysis (intersection) results also save to ''' filename_ternary '''']);

%% ternary surf plot for individual objectives
for i=1:length(ternary_obj_ids)
    fig_id = figure(i+10); 
    ax = axes(fig_id);
    set(fig_id,'Renderer','openGL');
    
    id = ternary_obj_ids(i);
    fig_id.Position = [100 100 1200 800];
    d = responses(opt_ind, id);
    if is_maximize(id) % if maximization of object, revert color map
        option.cmap = colormap(flipud(jet(2000))); 
    else
        option.cmap = colormap(jet(2000)); 
    end
    option.num_contours = 1; % single target contour line with arrow
%     option.num_contours = 4; % multiple contour lines with the target contour line in bold
    option.dec_place = decimal_places_contour_lines(ternary_obj_ids(i)); 
    option.obj_id = i; % to distinguish the arrow's direction
    [~,~,hcb]=tersurf(ws(:,1),ws(:,2),ws(:,3),d,option);
    terlabel(weight_str{ternary_obj_ids(1)}, weight_str{ternary_obj_ids(2)}, weight_str{ternary_obj_ids(3)});
    hcb.Label.String = obj_str{id};
    
    ax.SortMethod = 'childorder'; % 
    print('-dpng',[directory_name '/'  filename '_ternary_' method '_' num2str(i) '.png']);
end

%% combinining all three contour lines with arrows towards the optimals
fig_id = figure(1000); fig_id.Position = [100 100 1200 800]; hold on;
% decimal places for the values on the contour lines
option.dec_places = decimal_places_contour_lines(ternary_obj_ids(1:3));
option.legend_str = {[obj_str{ternary_obj_ids(1)} ', ' num2str(acceptable_percents(ternary_obj_ids(1))) '%'], ...
                     [obj_str{ternary_obj_ids(2)} ', ' num2str(acceptable_percents(ternary_obj_ids(2))) '%'], ...
                     [obj_str{ternary_obj_ids(3)} ', ' num2str(acceptable_percents(ternary_obj_ids(3))) '%']};
% for data tooltips and labels
option.var_str = var_str;
option.obj_str = obj_str;
option.variables = variables(opt_ind,:);
option.responses = responses(opt_ind,:);
option.weight_str = {weight_str{ternary_obj_ids(1:3)}};

% combined contour plot
option.is_arrow = false; % if set true, arrows direct towards optimal point will be added.
option.is_shade = true; % if set true, shade will be added to the intersection area.
tercontour(ws(:,1),ws(:,2),ws(:,3), d1, d2, d3, option);
print('-dpng',[directory_name '/' filename '_ternary_' method '_combined_contours.png']);


%% 3D plot using all points in the ternary plot
variables_opt = variables(opt_ind,:);
responses_opt = responses(opt_ind,:);

fig_id = figure(20); fig_id.Position = [100 100 1600 1200];
tile_id = tiledlayout(2,2);
txt = title(tile_id,'All points of Ternary Plot');
txt.FontSize = 18;
ax1 = nexttile;

scatter3(ax1, variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,1), 'filled');
colormap(ax1,jet); hcb = colorbar; hcb.Label.String = obj_str{1};
xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});

ax2 = nexttile;
scatter3(ax2, variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,2), 'filled');
colormap(ax2,jet); hcb = colorbar; hcb.Label.String = obj_str{2};
xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});

ax3 = nexttile;
scatter3(ax3, variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,3), 'filled');
colormap(ax3,flipud(jet));
hcb = colorbar; hcb.Label.String = obj_str{3};
xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});

axes_list = [ax1 ax2 ax3];
if num_objectives == 4
    ax4 = nexttile;
    scatter3(ax4, variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,4), 'filled');
    colormap(ax4,jet);  hcb = colorbar; hcb.Label.String = obj_str{4};
    xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});
    axes_list = [axes_list ax4];
end
print('-dpng',[directory_name '/' filename '_ternary_' method '_3D_plot_all_points.png']);
% to control multiple axies at the same time
hlink1 = linkprop(axes_list, {'CameraPosition','CameraUpVector'});

%% 3D plot using the points in the intersection of acceptable regions of the ternary plot
variables_opt = variables(opt_ind,:);
responses_opt = responses(opt_ind,:);

% ind_inter = responses_opt(:,ternary_obj_ids(1))<zval1 & responses_opt(:,ternary_obj_ids(2))<zval2 & responses_opt(:,ternary_obj_ids(3))<zval3;
variables_opt = variables_opt(ind_inter,:);
responses_opt = responses_opt(ind_inter,:);

fig_id = figure(21); fig_id.Position = [100 100 1600 1200];
tile_id = tiledlayout(2,2);
txt = title(tile_id,'Intersection points of Ternary Plot');
txt.FontSize = 18;
ax21 = nexttile;

scatter3(ax21,variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,1), 'filled');
colormap(ax21,jet); hcb = colorbar; hcb.Label.String = obj_str{1};
xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});

ax22 = nexttile;
scatter3(ax22,variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,2), 'filled');
colormap(ax22,jet); hcb = colorbar; hcb.Label.String = obj_str{2};
xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});

ax23 = nexttile;
scatter3(ax23,variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,3), 'filled');
colormap(ax23,flipud(jet));
hcb = colorbar; hcb.Label.String = obj_str{3};
xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});

axes_list = [ax21 ax22 ax23];
if num_objectives == 4
    ax24 = nexttile;
    scatter3(ax24,variables_opt(:,1), variables_opt(:,2), variables_opt(:,3), 30, responses_opt(:,4), 'filled');
    colormap(ax24,jet);  hcb = colorbar; hcb.Label.String = obj_str{4};
    xlabel(var_str{1}); ylabel(var_str{2}); zlabel(var_str{3});
    axes_list = [axes_list ax24];
end
print('-dpng',[directory_name '/' filename '_ternary_' method '_3D_plot_intersection_points.png']);

% to control multiple axies at the same time
hlink2 = linkprop(axes_list, {'CameraPosition','CameraUpVector'});

function distances = get_weighted_distance(responses, w)
% get the normalized distance with a weight of all points from the zero reference point.
min_val = min(responses);
range_val = max(responses) - min_val;
responses = (responses-min_val)./range_val;
distances = sum((responses.*w).^2,2);
