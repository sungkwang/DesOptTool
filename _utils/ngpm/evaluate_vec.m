function [pop, state] = evaluate_vec(opt, pop, state, varargin)
% Function: [pop, state] = evaluate(opt, pop, state, varargin)
% Description: Evaluate the objective functions of each individual in the
%   population.
%
%         LSSSSWC, NWPU
%    Revision: 1.0  Data: 2011-04-20
% sungkwang: modified for vectorization
%*************************************************************************

N = length(pop);
allTime = zeros(N, 1);  % allTime : use to calculate average evaluation times

%*************************************************************************
% Evaluate objective function in vector form
[pop, allTime] = eval_all(pop, opt.objfun, varargin{:});

%*************************************************************************
% Statistics
%*************************************************************************
state.avgEvalTime   = sum(allTime) / length(allTime);
state.evaluateCount = state.evaluateCount + length(pop);

function [pop, evalTime] = eval_all(pop, objfun, varargin)
% Function: [pop, evalTime] = evalIndividual(pop, objfun, varargin)
% Description: Evaluate one objective function.
% Sungkwang: modified to process all data in an array at once
%*************************************************************************

tStart = tic;
num_i = length(pop);
num_x = length(pop(1).var);
x_all = reshape([pop.var],[num_x, num_i]);
[y, cons] = objfun( x_all', varargin{:} );

% Save the objective values and constraint violations
for i = 1:num_i
    pop(i).obj = y(i,:);
    if( ~isempty(pop(i).cons) )
%         idx = find( cons(i,:) );
        idx = any(cons(i,:),1);
        pop(i).nViol = sum(idx);
        pop(i).violSum = sum( abs(cons(i,:)) );
%         if( ~isempty(idx) )
%             pop(i).nViol = length(idx);
%             pop(i).violSum = sum( abs(cons(i,:)) );
%         end
    end
end
evalTime = toc(tStart);