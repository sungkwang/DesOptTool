function [c, h] = tercontour(c1,c2,c3,d1,d2,d3, option)
%FUNCTION [HG,HTICK,HCB]=TERCONTOUR(C1,C2,C3,D1,D2,D3,VALUES) plots the values in the 
% vector d1, d2, d3 as contour plots of three objectives in a ternary diagram.
% The three vectors c1,c2,c3 define the position of a data value within the
% ternary diagram.
% The ternary axis system is created within the function.
% Axis label can be added using the terlabel function.
%
% Uli Theune, Geophysics, University of Alberta
% 2002 - ...
% modified by Sungkwang Mun for combined contour lines with arrows and shade on intersection area, sungkwan@cavs.msstate.edu
%
set(0,'DefaultAxesFontName', 'Times New Roman','DefaultAxesFontSize', 18)
set(0,'DefaultTextFontName', 'Times New Roman','DefaultTextFontSize', 18)

if nargin < 6
    error('Error: Not enough input arguments.');
end
if (length(c1)+length(c2)+length(c3))/length(c1) ~=3
    error('Error: all arrays must be of equal length.');
end

if ~isfield(option,'dec_places') % decimal place for the number values on contour lines
    option.dec_places = [2 2 2];
end

if ~isfield(option,'legend_str') % decimal place for the number values on contour lines
    option.legend_str = {'obj1', 'obj2', 'obj3'};
end

if ~isfield(option,'is_arrow') % decimal place for the number values on contour lines
    option.is_arrow = true;
end

if ~isfield(option,'is_shade') % if set true, shade will be added to the intersection area.
    option.is_shade = false;
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
zmax=max([max(d1) max(d2) max(d3)]);
[hg, htick] = add_ternary_axis(zmax);
axis image
axis off
% caxis([min(d) max(d)])
view(2)

% Calculate the position of the data points in the ternary diagram
x=0.5-c1*cos(pi/3)+c2/2;
y=0.866-c1*sin(pi/3)-c2*cot(pi/6)/2;

% label
terlabel(option.weight_str{1}, option.weight_str{2}, option.weight_str{3});

% contour plot
xg=linspace(0,1,51);
yg=linspace(0,0.866,51);
[X,Y]=meshgrid(xg,yg);

% contour line for obj1
color = 'r';
zval = option.zvals(1); dec_place = option.dec_places(1);
Z=griddata(x,y,d1,X,Y);
[c1,h]=contour(X,Y,Z, [zval zval],'ShowText','on','color',color);
clabel(c1,h,'FontSize',14); set(h,'linewidth',4);
h.LevelList = round(h.LevelList, dec_place);
pid(1) = h;

interval = 2;
% add arrows towards each objective
% obj1: [0 0], obj2: [0 1] obj3: [0.5 0.8660 (sqrt(3)/2)];
c1(:,abs(c1(1,:)-zval)<zval*1e-1) = [];
cline = c1(:,2:interval:end-1);

% obj1: [0 0]
ref_point = zeros(size(cline)); 
if option.is_arrow
    plot_arraw(cline, ref_point, color)
end

% contour line for obj2
color = 'g';
zval = option.zvals(2); dec_place = option.dec_places(2);
Z=griddata(x,y,d2,X,Y);
[c2,h]=contour(X,Y,Z, [zval zval],'ShowText','on','color',color);
clabel(c2,h,'FontSize',14); set(h,'linewidth',4);
h.LevelList = round(h.LevelList, dec_place);
pid(2) = h;

% obj2: [0 1]
c2(:,abs(c2(1,:)-zval)<zval*1e-1) = [];
cline = c2(:,2:interval:end-1);
ref_point = zeros(size(cline)); 
ref_point(1,:) = 1;
if option.is_arrow
    plot_arraw(cline, ref_point, color)
end

% contour line for obj3
color = 'b'; 
zval = option.zvals(3); dec_place = option.dec_places(3);
Z=griddata(x,y,d3,X,Y);
[c3,h]=contour(X,Y,Z, [zval zval],'ShowText','on','color',color);
clabel(c3,h,'FontSize',14); set(h,'linewidth',4);
h.LevelList = round(h.LevelList, dec_place);
pid(3) = h;

% obj3: [0.5 0.8660]
c3(:,abs(c3(1,:)-zval)<zval*1e-1) = [];
cline = c3(:,2:interval:end-1);
ref_point = zeros(size(cline)); 
ref_point(1,:) = 0.5;
ref_point(2,:) = 0.8660;
if option.is_arrow
    plot_arraw(cline, ref_point, color)
end

c11 = c1;
c22 = c2;
if norm(c22(:,1)) > norm(c22(:,end))
    c22 = c22(:,end:-1:1);
end
c33 = c3;

if option.is_shade 
    % find intersection points between c1 and c2 and c3
    c = [c11, nan(2,1), c33, nan(2,1), c22];
    [p, ind_inter] = InterX(c);

    if ~isempty(ind_inter)
        cind = zeros(3,length(ind_inter));
        for i = 1:length(ind_inter)
            %    ind = ind_inter(i);
            ind = find(sum(abs(c11-p(:,i)))<0.01);
            if ind, cind(1,i) = ind(1); end % if inds is more than 1, just choose the first one

            ind = find(sum(abs(c22-p(:,i)))<0.01);
            if ind, cind(2,i) = ind(1); end

            ind = find(sum(abs(c33-p(:,i)))<0.01);
            if ind, cind(3,i) = ind(1); end
        end
    end
    count = sum(cind~=0); % intersection point needs pair points.
    if all(count == 2)
        ind1_remove = [];
        ind2_remove = [];
        ind3_remove = [];
        for i = 1:length(ind_inter)
            inds = find(cind(:,i)~=0);
            if length(inds)==2
                if all(inds == [1 3]')
                    ind1_remove = [ind1_remove cind(1,i):length(c11)];
                    ind3_remove = [ind3_remove 1:cind(3,i)];
                elseif all(inds == [2 3]')
                    ind3_remove = [ind3_remove cind(3,i):length(c33)];
                    ind2_remove = [ind2_remove 1:cind(2,i)];
                elseif all(inds == [1 2]')
                    ind1_remove = [ind1_remove 1:cind(1,i)];
                    ind2_remove = [ind2_remove cind(2,i):length(c22)];
                else
                    error('not implemented yet');
                end
            end
        end
        c11(:,ind1_remove) = [];
        c33(:,ind3_remove) = [];
        c22(:,ind2_remove) = [];

        % final line set that includes the intersection region.
        c = [c11 c33 c22];
        patch(c(1,:),c(2,:),[.5 0 .5],'FaceAlpha',.3, 'LineStyle','none')

    %     for i=1:length(c)
    %         plot(c(1,i),c(2,i),'.y','MarkerSize',i);
    %     end
    end
end

% availble data points on the ternary diagram and their corresponding
% information regarding design variables and responses
% pid_data = plot3(x,y,zmax*1.1*ones(size(x)),'k.','MarkerSize',8);
pid_data = plot(x(option.ind_inter),y(option.ind_inter),'k.','MarkerSize',8);

hold off;
legend(pid, option.legend_str);

if all(isfield(option,{'variables','responses', 'var_str', 'obj_str'}))
    dtRows = [];
    for i=1:length(option.var_str)
         dtRows = [dtRows  dataTipTextRow(['In: ' option.var_str{i}],option.variables(:,i))];
    end
    for i=1:length(option.obj_str)
         dtRows = [dtRows  dataTipTextRow(['Out: ' option.obj_str{i}],option.responses(:,i))];
    end
    
%     dtRows = [dataTipTextRow(['In: ' option.var_str{1}],option.variables(:,1)),...
%               dataTipTextRow(['In: ' option.var_str{2}],option.variables(:,2)),...
%               dataTipTextRow(['In: ' option.var_str{3}],option.variables(:,3)),...
%               dataTipTextRow(['Out: ' option.obj_str{1}],option.responses(:,1)), ...
%               dataTipTextRow(['Out: ' option.obj_str{2}],option.responses(:,2)), ...
%               dataTipTextRow(['Out: ' option.obj_str{3}],option.responses(:,3)), ...
%               dataTipTextRow(['Out: ' option.obj_str{4}],option.responses(:,4))];
    pid_data.DataTipTemplate.DataTipRows = dtRows;
end


function plot_arraw(cline, ref_point, color)
uv = ref_point-cline;
q = quiver(cline(1,:),cline(2,:), uv(1,:),uv(2,:));
q.Color = color;
q.AutoScaleFactor = 0.3;
q.LineWidth = 1;
% q.ShowArrowHead = 'off';
% % q.Alignment = 'head';
% q.Marker = '>';

function [hg, htick] = add_ternary_axis(zmax)
% Add the axis system now
d1=cos(pi/3);
d2=sin(pi/3);
l=linspace(0,1,6);
for i=2:length(l)-1
%    hg(i-1,3)=plot3([l(i)*d1 1-l(i)*d1],[l(i)*d2 l(i)*d2],[zmax zmax]*1.1,':k','linewidth',0.25);
%    hg(i-1,1)=plot3([l(i) l(i)+(1-l(i))*d1],[0 (1-l(i))*d2],[zmax zmax]*1.1,':k','linewidth',0.25);
%    hg(i-1,2)=plot3([(1-l(i))*d1 1-l(i)],[(1-l(i))*d2 0],[zmax zmax]*1.1,':k','linewidth',0.25);
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

function clines = get_contour_lines(cnt)
% return controu lines after splitting contour vector from coutour function
szc = size(cnt);
idz = 1;
while idz<szc(2)
    izi = cnt(2,idz);
    cnt(2,idz) = nan;
    idz = idz+izi+1;
end
inds = find(isnan(cnt(2,:)));
clines = cell(1,length(inds));
for i=1:length(inds)-1
    clines{i} = cnt(:,inds(1):inds(2));
end
clines{length(inds)} = cnt(:,inds(end):end);



