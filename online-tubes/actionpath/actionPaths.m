% ---------------------------------------------------------
function actionPaths(dopts)
% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MID License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

detresultpath = dopts.detDir;
costtype = dopts.costtype;
gap = dopts.gap;
videolist = dopts.vidList;
actions = dopts.actions;
saveName = dopts.actPathDir;
iouth = dopts.iouThresh;
numActions = length(actions);
nms_thresh = 0.45;
videos = getVideoNames(videolist);
NumVideos = length(videos);

for vid=1:NumVideos
    tic;
    videoID  = videos{vid};
    pathsSaveName = [saveName,videoID,'-actionpaths.mat'];
    
    videoDetDir = [detresultpath,videoID,'/'];

    if ~exist(pathsSaveName,'file')
        fprintf('computing tubes for vide [%d out of %d] video ID = %s\n',vid,NumVideos, videoID);
        
        %% loop over all the frames of the video
        fprintf('Reading detections ');
        
        frames = readDetections(videoDetDir);
        
        fprintf('\nDone reading detections\n');
        
        fprintf('Gernrating action paths ...........\n');
        
        %% parllel loop over all action class and genrate paths for each class
        allpaths = cell(1);
        parfor a=1:numActions
            allpaths{a} = genActionPaths(frames, a, nms_thresh, iouth, costtype,gap);
        end
        
        fprintf('results are being saved in::: %s for %d classes\n',pathsSaveName,length(allpaths));
        save(pathsSaveName,'allpaths');
        fprintf('All Done in %03d Seconds\n',round(toc));
    end

end

disp('done computing action paths');

end

function paths = genActionPaths(frames,a,nms_thresh,iouth,costtype,gap)
action_frames = struct();

for f=1:length(frames)
    [boxes,scores,allscores] = dofilter(frames,a,f,nms_thresh);
    action_frames(f).boxes = boxes;
    action_frames(f).scores = scores;
    action_frames(f).allScores = allscores;
end

paths = incremental_linking(action_frames,iouth,costtype, gap, gap);

end

%-- filter out least likkey detections for actions ---
function [boxes,scores,allscores] = dofilter(frames, a, f, nms_thresh)
    scores = frames(f).scores(:,a);
    pick = scores>0.001;
    scores = scores(pick);
    boxes = frames(f).boxes(pick,:);
    allscores = frames(f).scores(pick,:);
    [~,pick] = sort(scores,'descend');
    to_pick = min(50,size(pick,1));
    pick = pick(1:to_pick);
    scores = scores(pick);
    boxes = boxes(pick,:);
    allscores = allscores(pick,:);
    pick = nms([boxes scores], nms_thresh);
    pick = pick(1:min(10,length(pick)));
    boxes = boxes(pick,:);
    scores = scores(pick);
    allscores = allscores(pick,:);
end

%-- list the files in directory and sort them ----------
function list = sortdirlist(dirname)
list = dir(dirname);
list = sort({list.name});
end

% -------------------------------------------------------------------------
function [videos] = getVideoNames(split_file)
% -------------------------------------------------------------------------
fprintf('Get both lis is %s\n',split_file);
fid = fopen(split_file,'r');
data = textscan(fid, '%s');
videos  = cell(1);
count = 0;

for i=1:length(data{1})
    filename = cell2mat(data{1}(i,1));
    count = count +1;
    videos{count} = filename;
    %     videos(i).vid = str2num(cell2mat(data{1}(i,1)));
end
end


function frames = readDetections(detectionDir)

detectionList = sortdirlist([detectionDir,'*.mat']);
frames = struct([]);
numframes = length(detectionList);
scores = 0;
loc = 0;
for f = 1 : numframes
  filename = [detectionDir,detectionList{f}];
  load(filename); % loads loc and scores variable
  loc = [loc(:,1)*320, loc(:,2)*240, loc(:,3)*320, loc(:,4)*240];
  loc(loc(:,1)<0,1) = 0;
  loc(loc(:,2)<0,2) = 0;
  loc(loc(:,3)>319,3) = 319;
  loc(loc(:,4)>239,4) = 239;
  loc = loc + 1;
  frames(f).boxes = loc;
  frames(f).scores = [scores(:,2:end),scores(:,1)];
end

end
