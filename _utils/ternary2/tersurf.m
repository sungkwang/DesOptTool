function [hg,htick,hcb]=tersurf(c1,c2,c3,d, option)
%FUNCTION [HG,HTICK,HCB]=TERSURF(C1,C2,C3,D) plots the values in the vector d 
% as a pseudo color plot in a ternary diagram.
% The three vectors c1,c2,c3 define the position of a data value within the
% ternary diagram.
% The ternary axis system is created within the function.
% Axis label can be added using the terlabel function.
% The function returns three handels: hg can be used to modify the grid lines,
% htick must be used to access the tick label properties, and hcb is the handle
% for the colorbar.
%
% Uli Theune, Geophysics, University of Alberta
% 2002 - ...
%
% modified by Sungkwang Mun to add contour plot, sungkwan@cavs.msstate.edu
%
set(0,'DefaultAxesFontName', 'Times New Roman','DefaultAxesFontSize', 18)
set(0,'DefaultTextFontName', 'Times New Roman','DefaultTextFontSize', 18)

if nargin < 4
    error('Error: Not enough input arguments.');
end
if (length(c1)+length(c2)+length(c3))/length(c1) ~=3
    error('Error: all arrays must be of equal length.');
end

if ~isfield(option,'num_contours') % number of contour lines
    option.num_contours = 4;
end

if ~isfield(option,'dec_place') % decimal place
    option.dec_place = 2;
end

% Check if the data need to be normalized
if max(c1+c2+c3)>1
    for i=1:length(c1)
        c1(i)=c1(i)/(c1(i)+c2(i)+c3(i));
        c2(i)=c2(i)/(c1(i)+c2(i)+c3(i));
        c3(i)=c3(i)/(c1(i)+c2(i)+c3(i));
    end
end

hold on

axis image; axis off;
% caxis([min(d) max(d)])
view(2)

% Calculate the position of the data points in the ternary diagram
x=0.5-c1*cos(pi/3)+c2/2;
y=0.866-c1*sin(pi/3)-c2*cot(pi/6)/2;

% Create short vectors for the griding
tri=delaunay(x,y);
% trisurf(tri,x,y,d);
trisurf(tri,x,y,d,'FaceAlpha',0.7);
shading interp

% Calculate the position of the data points in the ternary diagram
[hg, htick] = add_ternary_axis;

% contour plot
xg=linspace(0,1,51);
yg=linspace(0,0.866,51);
[X,Y]=meshgrid(xg,yg);
clear xg yg;
Z=griddata(x,y,d,X,Y);

zval = option.zvals(option.obj_id);
if option.num_contours > 1
    % a contour lines minus target contour line
    [c,h]=contour(X,Y,Z,option.num_contours-1,'ShowText','on','color','k');
    clabel(c,h,'FontSize',18) % set(h,'linewidth',4)
    h.LevelList = round(h.LevelList, option.dec_place);
    
    % target contour line in bold
    [c,h]=contour(X,Y,Z, [zval zval],'ShowText','on','color','k');
    clabel(c,h,'FontSize',18); 
    set(h,'linewidth',4)
    h.LevelList = round(h.LevelList, option.dec_place);
else
    % target contour line in bold
    [c,h]=contour(X,Y,Z, [zval zval],'ShowText','on','color','k');
    clabel(c,h,'FontSize',18); 
    set(h,'linewidth',4)
    h.LevelList = round(h.LevelList, option.dec_place);    
    
    % add arrow towards reference point 
    cline = c(:,2:4:end-1);
    cline(:,abs(cline(1,:)-zval)<zval*1e-3) = [];

    if option.obj_id == 1 % obj1: [0 0]
        ref_point = zeros(size(cline)); 
    elseif option.obj_id == 2 % obj2: [0 1]
        ref_point = zeros(size(cline)); 
        ref_point(1,:) = 1;
    elseif option.obj_id == 3 % obj3: [0.5 0.8660]
        ref_point = zeros(size(cline)); 
        ref_point(1,:) = 0.5;
        ref_point(2,:) = 0.8660;
    else
        error('obj_id must be between 1 and 3');
    end
    plot_arraw(cline, ref_point, 'k');
end

% Compute location of color bar pointer and make annotation:
hcb=colorbar;
barPos = get(hcb, 'Position');
set(hcb,'Position',[barPos(1)+0.05 barPos(2) barPos(3) barPos(4)]) % change position

% add arrow and acceptable percentage
cLimits = caxis();
barPos = get(hcb, 'Position');
xArrow = barPos(1)+ [-0.03 0];
yArrow = barPos(2)+barPos(4)*(zval-cLimits(1))/diff(cLimits)+[0 0];
val = (zval-min(d))/(max(d)-min(d))*100;
annotation('textarrow', xArrow, yArrow, 'String', [num2str(round(val)) '%'], ...
    'linewidth', 3, 'Headwidth',14, 'Headlength',14, 'Color', [0.6350 0.0780 0.1840]);

hold off

function plot_arraw(cline, ref_point, color)
uv = ref_point-cline;
q = quiver(cline(1,:),cline(2,:), uv(1,:),uv(2,:));
q.Color = color;
q.AutoScaleFactor = 0.3;
% q.LineWidth = 2;
% q.ShowArrowHead = 'off';
% % q.Alignment = 'head';
% q.Marker = '>';

function [hg, htick] = add_ternary_axis
% Add the axis system now
d1=cos(pi/3);
d2=sin(pi/3);
l=linspace(0,1,6);
for i=2:length(l)-1
   hg(i-1,3)=plot([l(i)*d1 1-l(i)*d1],[l(i)*d2 l(i)*d2],':k','linewidth',0.25);
   hg(i-1,1)=plot([l(i) l(i)+(1-l(i))*d1],[0 (1-l(i))*d2],':k','linewidth',0.25);
   hg(i-1,2)=plot([(1-l(i))*d1 1-l(i)],[(1-l(i))*d2 0],':k','linewidth',0.25);
end
plot([0 1 0.5 0],[0 0 sqrt(3)/2 0],'k','linewidth',1)
% Make x-tick labels
for i=1:length(l)
    htick(i,1)=text(l(i),-0.025,num2str(l(i)));
    htick(i,3)=text(1-l(i)*cos(pi/3)+0.025,l(i)*sin(pi/3)+0.025,num2str(l(i)));
    htick(i,2)=text(0.5-l(i)*cos(pi/3)-0.06,sin(pi/3)*(1-l(i)),num2str(l(i)));
end
