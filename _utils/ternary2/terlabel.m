function handels=terlabel(label1,label2,label3,option)
%FUNCTIONS HANDELS=TERLABEL(LABEL1,LABEL2,LABEL3) adds labels to a ternary 
% plot. Note that the order of labels must be the same as in the vectors in
% the ternaryc function call.
% The labels can be modified through the handel vector HANDELS.
%
% Uli Theune, Geophysics, University of Alberta
% 2005
%

if ~exist('option', 'var') || ~isfield(option,'colors') % decimal place
    option.colors = {'k', 'k', 'k'}; % default: black color
end

if nargout >= 1
    handels=ones(3,1);
    handels(1)=text(0.5,-0.05,label2,'horizontalalignment','center', 'Color', option.colors{1});
    handels(2)=text(0.15,sqrt(3)/4+0.05,label1,'horizontalalignment','center','rotation',60, 'Color', option.colors{2});
    handels(3)=text(0.85,sqrt(3)/4+0.05,label3,'horizontalalignment','center','rotation',-60, 'Color', option.colors{3});
else
    text(0.5,-0.05,label2,'horizontalalignment','center', 'Color', option.colors{1});
    text(0.15,sqrt(3)/4+0.05,label1,'horizontalalignment','center','rotation',60, 'Color', option.colors{2});
    text(0.85,sqrt(3)/4+0.05,label3,'horizontalalignment','center','rotation',-60, 'Color', option.colors{3});
end