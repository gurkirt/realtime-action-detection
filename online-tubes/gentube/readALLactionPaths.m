% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function actionpath = readALLactionPaths(videolist,actionPathDir,step)

videos = getVideoNames(videolist);
NumVideos = length(videos);

actionpath = struct([]);
fprintf('Loading action paths of %d videos\n',NumVideos);
count  = 1;
for vid=1:step:NumVideos
    
    videoID  = videos(vid).video_id;
    pathsSaveName = [actionPathDir,videoID,'-actionpaths.mat'];
   
    if ~exist(pathsSaveName,'file')
        error('Action path does not exist please genrate actin path', pathsSaveName)
    else
%         fprintf('loading vid %d %s \n',vid,pathsSaveName);
        load(pathsSaveName);
        actionpath(count).video_id = videos(vid).video_id;
        actionpath(count).paths = allpaths;
        count = count+1;
    end
end
end

function [videos] = getVideoNames(split_file)
% -------------------------------------------------------------------------
fid = fopen(split_file,'r');
data = textscan(fid, '%s');
videos  = struct();
for i=1:length(data{1})
    filename = cell2mat(data{1}(i,1));
    videos(i).video_id = filename;
    %     videos(i).vid = str2num(cell2mat(data{1}(i,1)));
    
end
count = length(data{1});

end
