function run_optimization_GA_part_design
% This function run genetic algorithm (NSGA-II) using trained NN model,
% Response surface method (RSM), or Kriging Method (KRG).
% 
% Authors/Contacts:
%    Sungkwang Mun <sungkwan@cavs.msstate.edu>
%

set(0,'DefaultAxesFontName', 'Times New Roman','DefaultAxesFontSize', 18)
set(0,'DefaultTextFontName', 'Times New Roman','DefaultTextFontSize', 18)
close all; clear;
addpath(genpath('_utils'));


% choose the surrogate approach below and modify the respective objective
% function at the end of this code accordingly. Keep in mind that number of
% constraints below should be changed change. 

method = 'ANN';
% method = 'RSM';
% method = 'KRG';

global  trained_model is_maximize is_ensemble is_single_net 
% copy the model location from the ANN training script to the model name
% variable below
if strcmp(method, 'ANN')
    % data/specification for ANN
    is_ensemble = false;
    is_single_net = false;
    %%% data/specification for ANN
    model_name = './trained_model/designA/trained_model_designA__hl_20_20_nd_100_tr_0.7_0_0.3_i_1234.mat';
    load(model_name, 'trained_model');
%     trained_model = trained_model;
end

%% name of variables and objectives
var_str = {'Length (mm)', 'Width (mm)', 'Thickness (mm)'};
obj_str = {'Mass (g)', 'Stress (MPa)', 'BucklingLoad (MPa)'};
obj_ids = [1 2 3]; % to plot the objective in particular order; x, y, and z.

number_of_constraints = 2; % number of constraints

var_lower_bounds = [24 3 0.3]; % lower bound of design variable
var_upper_bounds = [40 9 0.9]; % upper bound of design variable
variable_types = [1 1 1]; % variable type; 1: real, 2:integer

%% minimize/maximize indicator, e.g., [0 1 1]: 2nd and 3rd objective is maximized
% it will flip the sign of GA results whose objectives are to be maximized
is_maximize = [0 0 1]; % 0: minimize, 1: maximize
is_maximize = logical(is_maximize);

%% option for GA and visualization
is_from_scratch = false; % to run optimization again regardless of existence of result data
is_video_output = false; % to generate 3D video of the figure at the last iteration

num_iteration = 100;
num_population = 500;

num_variables = length(var_str);
num_objectives = length(obj_str);

directory_name = 'results/GA';
if ~exist(directory_name, 'dir'), mkdir(directory_name); end
filename = [method '_nPop_' num2str(num_population) '_nIter_' num2str(num_iteration)];
% load the previous searched optimization results
if exist([directory_name '/' filename '.mat'], 'file') && ~is_from_scratch
    load([directory_name '/' filename '.mat'], 'result_Optim', 'option');
    disp(['GA results file loaded from ' directory_name '/' filename '.mat']);
%     close(result_Optim.fig_handle); % to close the figure from the previous simulation.
else
    option = nsgaopt();                  % create default option structure
%     option.outputfile = ['populations_' filename '.txt'];
    option.is_maximize = is_maximize;
    option.popsize = num_population;               % populaion size
    option.maxGen  = num_iteration;                % max generation
    option.method = method;                        % surrogate modeling method
%     option.video_filename = filename;            % video output during iteration
    
    %% R-NSGA-II option for reference point method and normalization of the
    option.refUseNormDistance = 'front';
    option.refEpsilon = 0.01; % higher wider spread of points

    %% design variables and objectives and their bounds
    option.numVar = num_variables;            % number of design variables
    option.vartype = variable_types;    
%     option.decPlace = [4 4 4];              % decimal place for each design variable, e.g. 3.5123213 -> 3.512
    option.numObj = num_objectives;           % number of objectives
    option.numCons = number_of_constraints;   % number of constraints
    option.lb = var_lower_bounds;             % lower bound of variable
    option.ub = var_upper_bounds;             % upper bound of variable
    option.nameVar = var_str;                 % the variable names    
    option.nameObj = obj_str;                 % the objective names are showed in GUI window.    
    option.plotInterval = 10;                 % interval between two calls of "plotnsga". 
    
    % objective function handle
    if strcmp(method, 'ANN')
        option.objfun = @ANN_objfun;         
    elseif strcmp(method, 'RSM')
        option.objfun = @RSM_objfun;    
    elseif strcmp(method, 'KRG')
        option.objfun = @KRG_objfun;    
    else
        error(['Unkown objective function' option.objfun]);
    end
    result_Optim = nsga2_vec(option);         % begin the optimization!
    
    result_Optim = rmfield(result_Optim,'fig_handle'); % remove figure handle which is unnecessary.
    save([directory_name '/' filename '.mat'],'result_Optim','option') % save the optimization results
    disp(['GA results file saved to ' directory_name '/' filename '.mat']);
end

%% Prepare data, axis labels for x,y, and z axis and c for color code
% get final pareto front responses/objectives and variables from GA
final_gen = result_Optim.pops(option.maxGen,:);
responses = reshape([final_gen.obj], [option.numObj option.popsize])';
variables = reshape([final_gen.var], [option.numVar option.popsize])';

% matrix plot
fig_id = figure(99);  fig_id.Position = [100 100 1200 800];
[~,ax]=plotmatrix(responses); 
for i = 1:length(obj_str)
    ax(i,1).YLabel.String = obj_str{i}; 
    ax(length(obj_str),i).XLabel.String = obj_str{i}; 
end
print('-dpng',[directory_name '/' filename '_' method '_matrix_plot.png']);

%% plot the final Pareto Front
% plots for all response values of final Pareto front and the extrema of every objective and "unweighted" optimum.
% it also shows correponding variables in tooltop if the point is clicked.
fig_id = figure(1); fig_id.Position = [100 100 1200 800];
option.method = method;
option.obj_ids = obj_ids;
option.var_str = var_str;
option.obj_str = obj_str;
plot_pareto_front(variables, responses, option);
print('-dpng', [directory_name '/' filename '_Pareto_front_' method '.png']);

% For video animation of the final pareto front. Useful for 3D and 4D.
if is_video_output
    OptionZ.FrameRate=120;
    OptionZ.Duration=8;
    OptionZ.Periodic=false;
%     view_list = [-20,10; -110,10; -190,80; -290,10;-380,10];
    view_list = [-20,10; -110,10; -190,80;];
    CaptureFigVid(view_list, [directory_name '/' filename '_final_iter'], OptionZ)
end

function [y, cons] = ANN_objfun(x)
% artificial neural network (ANN) regression
% input:  x(1): length x(2): width x(3): thickness
% output: y(1): Mass y(2):Stress y(3): Buckling Load
%****************************************************
global trained_model is_maximize is_ensemble is_single_net 

option.is_single_net = is_single_net;
if is_ensemble
    y = predict_nets_ensemble(x', trained_model, option);
else
    y = predict_nets(x', trained_model,  option);
end
y = y';

% constrains on buckling load to be inbetween 200 MPa and 300 MPa.
cons = zeros(size(x,1),2);
c = y(:,3) - 200;
inds1 = c<0;
cons(inds1,1) = abs(c(inds1));

c = y(:,3) - 300;
inds2 = c>0;
cons(inds2,2) = abs(c(inds2));

% inds1 = y(:,3)<200;
% cons(inds1,1) = 1;
% inds2 = y(:,3)>300;
% ind = inds1 | inds2;
% cons(ind,1) = 1;

% inverse the sign of objective for maximization.
y(:,is_maximize) = -y(:,is_maximize);

function [y, cons] = KRG_objfun(x)
% Kriging regression
% input:  x(1): length x(2): width x(3): thickness
% output: y(1): Mass y(2):Stress y(3): Buckling Load
%****************************************************
global trained_model is_maximize

y = srgtsKRGPredictor(x, trained_model);

% constrains on buckling load to be inbetween 200 MPa and 300 MPa.
cons = zeros(size(x,1),2);
c = y(:,3) - 200;
inds1 = c<0;
cons(inds1,1) = abs(c(inds1));

c = y(:,3) - 300;
inds2 = c>0;
cons(inds2,2) = abs(c(inds2));

% inverse the sign of objective for maximization.
y(:,is_maximize) = -y(:,is_maximize);

function [y, cons] = RSM_objfun(x)
% Objective function for RSM
% input:  x(1): length x(2): width x(3): thickness
% output: y(1): Mass y(2):Stress y(3): Buckling Load
%****************************************************
global is_maximize
% y(:,1) = -3073.09 + 69.841*x(:,1) + 8063.188*x(:,2) + 1766.291*x(:,3);
% y(:,2) = 21.75 - 0.02658*x(:,1) - 25.3774*x(:,2) - 6.09895*x(:,3);
% y(:,3) = 7*(x(:,1)-14.8)+96;
% y(:,4) = 13.6397-0.12502*x(:,1)-6.85431*x(:,2)-1.46794*x(:,3);

% constrains on buckling load to be inbetween 200 MPa and 300 MPa.
cons = zeros(size(x,1),2);
c = y(:,3) - 200;
inds1 = c<0;
cons(inds1,1) = abs(c(inds1));

c = y(:,3) - 300;
inds2 = c>0;
cons(inds2,2) = abs(c(inds2));

% inverse the sign of objective for maximization.
y(:,is_maximize) = -y(:,is_maximize);
