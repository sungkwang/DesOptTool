function generate_data

if ~nargin
    close all
    is_plot = true;
    sampling_type = 'random';
%     sampling_type = 'lhs';
    num_data = 80;
    foldername = '_data/';
end

lb = [ 6  20  78 432];                 % lower bound of x
ub = [11 22.5 80 600];                 % upper bound of x


if strcmp(sampling_type, 'random')
    % random sampling
    x = (ub-lb).*rand(num_data,4) + lb;
elseif strcmp(sampling_type, 'lhs')
    % latine hypercube sampling
    x = lhsdesign_modified(num_data,lb, ub);
end

y_all = zeros(num_data,2);
for i = 1:num_data
    [y, cons] = TrapPrism_objfun(x(i,:));
    y_all(i,:) = y';
end

filename = [foldername sampling_type '_' num2str(num_data) '.mat'];
save(filename,'x','y_all');

if is_plot 
    % histogram plot of distribution of each variable.
    figure(1);
    subplot(2,2,1);
    histogram(x(:,1));
    title('d1');
    subplot(2,2,2);
    histogram(x(:,2));
    title('d2');
    subplot(2,2,3);
    histogram(x(:,3));
    title('h');
    subplot(2,2,4);
    histogram(x(:,4));
    title('wl');
    filename = [foldername sampling_type '_hist.png'];
    print('-dpng', filename);

    % output data.
    figure(2);
    plot(y_all(:,1),y_all(:,2),'o')
    xlabel('MoI');
    ylabel('Weight');
    grid on;
    filename = [foldername sampling_type '_Weight_vs_MoI.png'];
    print('-dpng', filename);
end

function [y, cons] = TrapPrism_objfun(x)
% Objective function : Test problem 'CONE'.
% input:  x(1):d1 x(2):d2  x(3):h  x(4):wl  
% output: y(1):I  y(2):W
%****************************************************
y = [0, 0];
cons = [0, 0];

y(1) = ((((x(1)+2*x(2))*x(4)*x(3)*0.00148)*((1/12)*x(1)^2 + (1/3)*x(1)*x(2) + (1/3)*x(2)^2 + (1/3)*x(3)^2)) - (2*((1/36)*(0.5*x(2)*x(4)*x(3)*0.00148)*(x(2)^2 + x(3)^2) + (0.5*x(2)*x(4)*x(3)*0.00148)*((x(3)/2)^2 + (((2*x(2))/3) + (x(1)/2))^2))));

%y(1) = -((((x(1)+2*x(2))*x(4)*x(3)*0.00148)*((1/12)*x(1)^2 + (1/3)*x(1)*x(2) + (1/3)*x(2)^2 + (1/3)*x(3)^2)) - 2*((1/36)*(0.5*x(2)*x(4)*x(3)*0.00148)*(x(2)^2 + x(3)^2) + (0.5*x(2)*x(4)*x(3)*0.00148)*((x(3)/2)^2 + (((2*x(2))/3) + (x(1)/2))^2)));
y(2) = 0.5*(2*x(1)+2*x(2))*x(4)*x(3)*0.00148*32.174;

% calculate the constraint violations
c = x(4)-432;
if(c<0)
    cons(1) = abs(c);
end
c = x(1) + 2*x(2) - 54;
if(c<0)
    cons(2) = abs(c);
end