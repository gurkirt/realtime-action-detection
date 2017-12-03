% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------


function createdires(basedirs,actions)
for s = 1: length(basedirs)
    savename = basedirs{s};
    for action = actions
        saveNameaction = [savename,action{1}];
        if ~isdir(saveNameaction)
            mkdir(saveNameaction);
        end
    end
end
end