function [X_scaled,X_normalized]=lhsdesign_modified(n,min_ranges_p,max_ranges_p)
%lhsdesign_modified is a modification of the Matlab Statistics function lhsdesign.
%It might be a good idea to jump straight to the example to see what does
%this function do.
%The following is the description of lhsdesign from Mathworks documentation
% X = lhsdesign(n,p) returns an n-by-p matrix, X, containing a latin hypercube sample of n values on each of p variables.
%For each column of X, the n values are randomly distributed with one from each interval (0,1/n), (1/n,2/n), ..., (1-1/n,1), and they are randomly permuted.

%lhsdesign_modified provides a latin hypercube sample of n values of
%each of p variables but unlike lhsdesign, the variables can range between
%any minimum and maximum number specified by the user, where as lhsdesign
%only provide data between 0 and 1 which might not be very helpful in many
%practical problems where the range is not bound to 0 and 1
%
%Inputs: 
%       n: number of radomly generated data points
%       min_ranges_p: [1xp] or [px1] vector that contains p values that correspond to the minimum value of each variable
%       max_ranges_p: [1xp] or [px1] vector that contains p values that correspond to the maximum value of each variable
%Outputs
%       X_scaled: [nxp] matrix of randomly generated variables within the
%       min/max range that the user specified
%       X_normalized: [nxp] matrix of randomly generated variables within the
%       0/1 range 
%
%Example Usage: 
%       [X_scaled,X_normalized]=lhsdesign_modified(100,[-50 100 ],[20  300]);
%       figure
%       subplot(2,1,1),plot(X_scaled(:,1),X_scaled(:,2),'*')
%       title('Random Variables')
%       xlabel('X1')
%       ylabel('X2')
%       grid on
%       subplot(2,1,2),plot(X_normalized(:,1),X_normalized(:,2),'r*')
%       title('Normalized Random Variables')
%       xlabel('Normalized X1')
%       ylabel('Normalized X2')
%       grid on


p=length(min_ranges_p);
[M,N]=size(min_ranges_p);
if M<N
    min_ranges_p=min_ranges_p';
end
    
[M,N]=size(max_ranges_p);
if M<N
    max_ranges_p=max_ranges_p';
end

slope=max_ranges_p-min_ranges_p;
offset=min_ranges_p;

SLOPE=ones(n,p);
OFFSET=ones(n,p);

for i=1:p
    SLOPE(:,i)=ones(n,1).*slope(i);
    OFFSET(:,i)=ones(n,1).*offset(i);
end
X_normalized = lhsdesign2(n,p);

X_scaled=SLOPE.*X_normalized+OFFSET;

function X = lhsdesign2(n,p)
%LHSDESIGN2 Generate a latin hypercube sample.

maxiter=5; crit='maximin'; dosmooth='on';

% Start with a plain lhs sample over a grid
X = getsample(n,p,dosmooth);

% Create designs, save best one
if isequal(crit,'none') || size(X,1)<2
    maxiter = 0;
end
switch(crit)
 case 'maximin'
   bestscore = score(X,crit);
   for j=2:maxiter
      x = getsample(n,p,dosmooth);
      
      newscore = score(x,crit);
      if newscore > bestscore
         X = x;
         bestscore = newscore;
      end
   end
 case 'correlation'
   bestscore = score(X,crit);
   for iter=2:maxiter
      % Forward ranked Gram-Schmidt step:
      for j=2:p
         for k=1:j-1
            z = takeout(X(:,j),X(:,k));
            X(:,k) = (rank(z) - 0.5) / n;
         end
      end
      % Backward ranked Gram-Schmidt step:
      for j=p-1:-1:1
         for k=p:-1:j+1
            z = takeout(X(:,j),X(:,k));
            X(:,k) = (rank(z) - 0.5) / n;
         end
      end
   
      % Check for convergence
      newscore = score(X,crit);
      if newscore <= bestscore
         break;
      else
         bestscore = newscore;
      end
   end
end

% ---------------------
function x = getsample(n,p,dosmooth)
x = rand(n,p);
for i=1:p
   x(:,i) = rank(x(:,i));
end
   if isequal(dosmooth,'on')
      x = x - rand(size(x));
   else
      x = x - 0.5;
   end
   x = x / n;
   
% ---------------------
function s = score(x,crit)
% compute score function, larger is better

if size(x,1)<2
    s = 0;       % score is meaningless with just one point
    return
end

switch(crit)
 case 'correlation'
   % Minimize the sum of between-column squared correlations
   c = corrcoef(x);
   s = -sum(sum(triu(c,1).^2));

 case 'maximin'
   % Maximize the minimum point-to-point difference
%    [~,dist] = knnsearch(x,x,'k',2);
%     s = min(dist(:,2));
   [~,dist] = knnsearch(x,x);
   s = min(dist);
 
end

% ------------------------
function z=takeout(x,y)

% Remove from y its projection onto x, ignoring constant terms
xc = x - mean(x);
yc = y - mean(y);
b = (xc-mean(xc))\(yc-mean(yc));
z = y - b*xc;

% -----------------------
function r=rank(x)

% Similar to tiedrank, but no adjustment for ties here
[~, rowidx] = sort(x);
r(rowidx) = 1:length(x);
r = r(:);


function [idx,D]=knnsearch(varargin)
% KNNSEARCH   Linear k-nearest neighbor (KNN) search
% IDX = knnsearch(Q,R,K) searches the reference data set R (n x d array
% representing n points in a d-dimensional space) to find the k-nearest
% neighbors of each query point represented by eahc row of Q (m x d array).
% The results are stored in the (m x K) index array, IDX. 
%
% IDX = knnsearch(Q,R) takes the default value K=1.
%
% IDX = knnsearch(Q) or IDX = knnsearch(Q,[],K) does the search for R = Q.
%
% Rationality
% Linear KNN search is the simplest appraoch of KNN. The search is based on
% calculation of all distances. Therefore, it is normally believed only
% suitable for small data sets. However, other advanced approaches, such as
% kd-tree and delaunary become inefficient when d is large comparing to the
% number of data points. On the other hand, the linear search in MATLAB is
% relatively insensitive to d due to the vectorization. In  this code, the 
% efficiency of linear search is further improved by using the JIT
% aceeleration of MATLAB. Numerical example shows that its performance is
% comparable with kd-tree algorithm in mex.
%
% See also, kdtree, nnsearch, delaunary, dsearch
% By Yi Cao at Cranfield University on 25 March 2008
% Example 1: small data sets
%{
R=randn(100,2);
Q=randn(3,2);
idx=knnsearch(Q,R);
plot(R(:,1),R(:,2),'b.',Q(:,1),Q(:,2),'ro',R(idx,1),R(idx,2),'gx');
%}
% Example 2: ten nearest points to [0 0]
%{
R=rand(100,2);
Q=[0 0];
K=10;
idx=knnsearch(Q,R,10);
r=max(sqrt(sum(R(idx,:).^2,2)));
theta=0:0.01:pi/2;
x=r*cos(theta);
y=r*sin(theta);
plot(R(:,1),R(:,2),'b.',Q(:,1),Q(:,2),'co',R(idx,1),R(idx,2),'gx',x,y,'r-','linewidth',2);
%}
% Example 3: cputime comparion with delaunay+dsearch I, a few to look up
%{
R=randn(10000,4);
Q=randn(500,4);
t0=cputime;
idx=knnsearch(Q,R);
t1=cputime;
T=delaunayn(R);
idx1=dsearchn(R,T,Q);
t2=cputime;
fprintf('Are both indices the same? %d\n',isequal(idx,idx1));
fprintf('CPU time for knnsearch = %g\n',t1-t0);
fprintf('CPU time for delaunay  = %g\n',t2-t1);
%}
% Example 4: cputime comparion with delaunay+dsearch II, lots to look up
%{
Q=randn(10000,4);
R=randn(500,4);
t0=cputime;
idx=knnsearch(Q,R);
t1=cputime;
T=delaunayn(R);
idx1=dsearchn(R,T,Q);
t2=cputime;
fprintf('Are both indices the same? %d\n',isequal(idx,idx1));
fprintf('CPU time for knnsearch = %g\n',t1-t0);
fprintf('CPU time for delaunay  = %g\n',t2-t1);
%}
% Example 5: cputime comparion with kd-tree by Steven Michael (mex file) 
% <a href="http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=7030&objectType=file">kd-tree by Steven Michael</a> 
%{
Q=randn(10000,10);
R=randn(500,10);
t0=cputime;
idx=knnsearch(Q,R);
t1=cputime;
tree=kdtree(R);
idx1=kdtree_closestpoint(tree,Q);
t2=cputime;
fprintf('Are both indices the same? %d\n',isequal(idx,idx1));
fprintf('CPU time for knnsearch = %g\n',t1-t0);
fprintf('CPU time for delaunay  = %g\n',t2-t1);
%}
% Check inputs
[Q,R,K,fident] = parseinputs(varargin{:});
% Check outputs
error(nargoutchk(0,2,nargout));
% C2 = sum(C.*C,2)';
[N,M] = size(Q);
L=size(R,1);
idx = zeros(N,K);
D = idx;
if K==1
    % Loop for each query point
    for k=1:N
        d=zeros(L,1);
        for t=1:M
            d=d+(R(:,t)-Q(k,t)).^2;
        end
        if fident
            d(k)=inf;
        end
        [D(k),idx(k)]=min(d);
    end
else
    for k=1:N
        d=zeros(L,1);
        for t=1:M
            d=d+(R(:,t)-Q(k,t)).^2;
        end
        if fident
            d(k)=inf;
        end
        [s,t]=sort(d);
        idx(k,:)=t(1:K);
        D(k,:)=s(1:K);
    end
end
if nargout>1
    D=sqrt(D);
end
function [Q,R,K,fident] = parseinputs(varargin)
% Check input and output
error(nargchk(1,3,nargin));
Q=varargin{1};
if nargin<2
    R=Q;
    fident = true;
else
    fident = false;
    R=varargin{2};
end
if isempty(R)
    fident = true;
    R=Q;
end
if ~fident
    fident = isequal(Q,R);
end
if nargin<3
    K=1;
else
    K=varargin{3};
end