% ---------------------------------------------------------
% Original code comes from  https://team.inria.fr/perception/research/skeletalquads/
% Copyright (c) 2014, Georgios Evangelidis and Gurkirt Singh,
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% M = <10xnum_frames>
% r = 10 (action labels)
% c = frame indices in a video

function [p,q,D] = dpEM_max(M,alpha)

% transition cost for the smoothness term
% V(L1,L2) = 0, if L1=L2
% V(L1,L2) = alpha, if L1~=L2



[r,c] = size(M);



% costs
D = zeros(r, c+1); % add an extra column
D(:,1) = 0; % put the maximum cost
D(:, 2:(c+1)) = M;

v = [1:r]';


%D = M;
phi = zeros(r,c);

%test = struct([]);
for j = 2:c+1; % c = 1230
    for i = 1:r; % r = 10        
        
%         test(j).D =  D(:, j-1); % fetching prev column 10 rows
%         test(j).alpha = alpha*(v~=i);  % switching each row for each class
%         test(j).D_alpha = [D(:, j-1)-alpha*(v~=i)];
%         test(j).max = max([D(:, j-1)-alpha*(v~=i)]); % for ith class taking the max score
        
        
        [dmax, tb] = max([D(:, j-1)-alpha*(v~=i)]);
        %keyboard;
        D(i,j) = D(i,j)+dmax;
        phi(i,j-1) = tb;
    end
end

% Note:
% the outer loop (j) is to visit one by one each frames
% the inner loop (i) is to get the max score for each action label
% the -alpha*(v~=i) term is to add a penalty by subtracting alpha from the 
% data term for all other class labels other than i, for ith class label 
% it adds zero penalty;
%  (v~=i) will return a logical array consists of 10 elements, in the ith 
% location it is 0 (false becuase the condition v~=i is false) and all other locations
% returns 1, thus for ith calss it multiplies 0
% with alpha and for the rest of the classes multiplies 1;
% for each iteration of ith loop we get a max value which we add to the
% data term d(i,j), in this way the 10 max values for 10 different action
% labels are stored to the jth column (or for the jth frame): D(1,j), D(2,j),...,D(10,j), 

%  save('test.mat','r','c','M', 'phi');
%  pause;

% Traceback from last frame
D = D(:,2:(c+1));

% best of the last column
q = c; % frame inidces
[~,p] = max(D(:,c));



i = p; % index of max element in last column of D, 
j = q; % frame indices

while j>1 % loop over frames in a video
    tb = phi(i,j); % i -> index of max element in last column of D, j-> last frame index or last column of D
    p = [tb,p];
    q = [j-1,q];
    j = j-1;
    i = tb;
end

%
% phi(i,j) stores all the max indices in the forward pass
% during the backward pass , a predicited path is constructed using these indices values
