function newpop = selectOp(opt, pop)
% Function: newpop = selectOp(opt, pop)
% Description: Selection operator, use binary tournament selection.
%
%         LSSSSWC, NWPU
%    Revision: 1.1  Data: 2011-07-12
% Sugnkwang: modified to have populations with specific floating point values.
% Sugnkwang: vectorized for speed-up (~30% faster)
%*************************************************************************

popsize = length(pop);
randnum = randi(popsize, [1, 2 * popsize]);

% pool = zeros(1, popsize);   % pool : the individual index selected
% j = 1;
% for i = 1:2:(2*popsize)
%     p1 = randnum(i);
%     p2 = randnum(i+1);
%     
%     if(~isempty(opt.refPoints))
%         % Preference operator (R-NSGA-II)
%         result = preferenceComp( pop(p1), pop(p2) );
%     else
%         % Crowded-comparison operator (NSGA-II)
%         result = crowdingComp( pop(p1), pop(p2) );
%     end
%     
%     if(result == 1)
%         pool(j) = p1;
%     else
%         pool(j) = p2;
%     end
%     
%     j = j + 1;
% end
% newpop = pop( pool );

p1 = randnum(1:2:end);
p2 = randnum(2:2:end);
if(~isempty(opt.refPoints))  % Preference operator (R-NSGA-II)
%     result = preferenceComp( pop(p1), pop(p2) );
    mask = [pop(p1).rank]<[pop(p2).rank] | [pop(p1).rank]==[pop(p2).rank] & [pop(p1).prefDistance]<[pop(p2).prefDistance];
else     % Crowded-comparison operator (NSGA-II)
%     result = crowdingComp( pop(p1), pop(p2) );
    mask = [pop(p1).rank]<[pop(p2).rank] | [pop(p1).rank]==[pop(p2).rank] & [pop(p1).distance]>[pop(p2).distance];
end
pool2 = zeros(1, popsize);
pool2(mask) = p1(mask);
pool2(~mask) = p2(~mask);
newpop = pop( pool2 );


function result = crowdingComp( guy1, guy2)
% Function: result = crowdingComp( guy1, guy2)
% Description: Crowding comparison operator.
% Return: 
%   1 = guy1 is better than guy2
%   0 = other cases
%
%         LSSSSWC, NWPU
%    Revision: 1.0  Data: 2011-04-20
%*************************************************************************

if((guy1.rank < guy2.rank) || ((guy1.rank == guy2.rank) && (guy1.distance > guy2.distance) ))
    result = 1;
else
    result = 0;
end



function result = preferenceComp(guy1, guy2)
% Function: result = preferenceComp(guy1, guy2)
% Description: Preference operator used in R-NSGA-II
% Return: 
%   1 = guy1 is better than guy2
%   0 = other cases
%
%    Copyright 2011 by LSSSSWC
%    Revision: 1.0  Data: 2011-07-11
%*************************************************************************

if(  (guy1.rank  < guy2.rank) || ...
    ((guy1.rank == guy2.rank) && (guy1.prefDistance < guy2.prefDistance)) )
    result = 1;
else
    result = 0;
end







