function pid = plot_pareto_front(variables, responses, option)
% This function plots responses and the extrema of all objectives and a "unweighted" optimum.
% It also shows correponding variables in tooltop if the point is clicked.
% If option.obj_ids is used, it can plot in axis with a specific order.
% i.e., obj(1):x axis, obj(2):y axis, obj(3): z axis, obj(4) if exist: color code
% This function can handle 2 to 4 objectives

method = option.method;
var_str = option.var_str;
obj_str = option.obj_str;
is_maximize = option.is_maximize;
if isfield(option,'obj_ids')
   obj_ids = option.obj_ids;
%    obj_str = obj_str(obj_ids);
else % if no obj_ids, just use the the ascending order
   obj_ids = 1:length(obj_str);
end
num_objectives = length(obj_ids); % for plotting purpose
if num_objectives < 2
    error('Unable to plot for the number of objectives less than 2');
elseif num_objectives > 4
    error('Unable to plot for the number of objectives greater than 4');
end

% rearrange the order of target objectives and associated strings for visualization
obj_str = obj_str(obj_ids);
res = responses(:, obj_ids); 
is_maximize = is_maximize(obj_ids);

% get distance between each objective to zero in normalized space. 
distances = get_distance(res);
% for unweighted optimum
[~, opt_ind] = min(distances); 
% for extrema
min_ind = zeros(1,num_objectives);
for i=1:num_objectives
    [~, min_ind(i)] = min(res(:,i));
end

% So far we inverted the sign for maximization to use the same minimization
% technique, so we revert the sign back here for the proper visualtion.
res(:,is_maximize) = -res(:,is_maximize);

hold on;
red = [0.6350 0.0780 0.1840];
% scatter plot for all points
if num_objectives == 2
    pid(1) = scatter(res(:,1), res(:,2),30, res(:,2), 'filled');
    c = colorbar;
    c.Label.String = obj_str{2};
    if is_maximize(2), colormap(flipud(jet)); end % invert the colormap if maxmize
    xlabel(obj_str{1}); ylabel(obj_str{2}); 
    pid(2) = plot(res(opt_ind,1), res(opt_ind,2),'p', 'Color', red, 'MarkerSize', 24, 'LineWidth', 4);
    for i = 1:num_objectives
        pid(i+2) = plot(res(min_ind(i),1), res(min_ind(i),2), 's','Color', red, 'MarkerSize', 24, 'LineWidth', 4);
    end    
elseif num_objectives == 3
    pid(1) = scatter3(res(:,1), res(:,2), res(:,3), 30, res(:,3), 'filled');
    c = colorbar;
    c.Label.String = obj_str{3};
    if is_maximize(3), colormap(flipud(jet)); end % invert the colormap if maxmize
    view(3);
    xlabel(obj_str{1}); ylabel(obj_str{2}); zlabel(obj_str{3});
    pid(2) = plot3(res(opt_ind,1), res(opt_ind,2),  res(opt_ind,3),'p', 'Color', red, 'MarkerSize', 24, 'LineWidth', 4);
    for i = 1:num_objectives
        pid(i+2) = plot3(res(min_ind(obj_ids(i)),1), res(min_ind(obj_ids(i)),2), res(min_ind(obj_ids(i)),3), 's','Color', red, 'MarkerSize', 24, 'LineWidth', 4);
    end    
elseif num_objectives == 4
    pid(1) = scatter3(res(:,1), res(:,2), res(:,3), 30, res(:,4), 'filled');
    c = colorbar;
    c.Label.String = obj_str{4};
    if is_maximize(4), colormap(flipud(jet)); end % invert the colormap if maxmize
    view(3);
    xlabel(obj_str{1}); ylabel(obj_str{2}); zlabel(obj_str{3});
    pid(2) = plot3(res(opt_ind,1), res(opt_ind,2),  res(opt_ind,3),'p', 'Color', red, 'MarkerSize', 24, 'LineWidth', 4);
    for i = 1:num_objectives
        pid(i+2) = plot3(res(min_ind(obj_ids(i)),1), res(min_ind(obj_ids(i)),2), res(min_ind(obj_ids(i)),3), 's','Color', red, 'MarkerSize', 24, 'LineWidth', 4);
    end
end
legend(pid(1:3),'Pareto front', 'Optimum', 'Extreme');
hold off; grid on; 

fprintf('**Learning type: %s\n', method);
for i=1:num_objectives
    response_vals = res(min_ind(i),:);
    if is_maximize(i)
        fprintf('Maximum %s: (%.2f, %.2f, %.2f, %.2f) <-', obj_str{i}, response_vals);
    else
        fprintf('Minimum %s: (%.2f, %.2f, %.2f, %.2f) <-', obj_str{i}, response_vals);
    end
    fprintf('(%.2f, %.2f, %.2f)\n', variables(min_ind(i),:));
end

% to change data tooltip for input variable and output response values
for i=1:length(pid)
    if i==1
        inds = 1:size(variables,1);
    elseif i==2
        inds = opt_ind;
    elseif i==3 
        inds = min_ind(1);
    elseif i==4 
        inds = min_ind(2);
    elseif i==5
        inds = min_ind(3);
    end

    dtRows = [];
    for j=1:length(var_str)
         dtRows = [dtRows  dataTipTextRow(['In: ' var_str{j}], variables(:,j))];
    end
    for j=1:length(obj_str)
         dtRows = [dtRows  dataTipTextRow(['Out: ' obj_str{j}], res(:,j))];
    end
    pid(i).DataTipTemplate.DataTipRows = dtRows;
end

function distances = get_distance(responses)
% get the normalized distance of every points from the zero reference point.
min_val = min(responses);
range_val = max(responses) - min_val;
responses = (responses-min_val)./range_val;
distances = sum(responses.^2,2);