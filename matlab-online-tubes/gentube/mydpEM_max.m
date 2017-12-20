function [p,q,D] = mydpEM_max(M,alpha)

[r,c] = size(M);
% costs
D = zeros(r, c+1); % add an extra column
D(:,1) = 0; % put the maximum cost
D(:, 2:(c+1)) = M;

v = [1:r]';
phi = zeros(r,c);

for j = 2:c+1; % c = 1230
    for i = 1:r; % r = 10        
        
        [dmax, tb] = max([D(:, j-1)-alpha*(v~=i)]);
        %keyboard;
        D(i,j) = D(i,j)+dmax;
        phi(i,j-1) = tb;
    end
end

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