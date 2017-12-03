function fusedActionPaths(dopts)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2016, Gurkirt Singh
%
% This code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

detresultpathBase = dopts.basedetDir;
detresultpathTop = dopts.topdetDir;
videolist = dopts.vidList;
actions = dopts.actions;
saveName = dopts.actPathDir;
iouth = dopts.iouThresh;
numActions = length(actions);
costtype = dopts.costtype;
gap = dopts.gap;
nms_thresh = 0.45;
videos = getVideoNames(videolist);

NumVideos = length(videos);
timimngs = zeros(NumVideos,1);

for vid=1:NumVideos
    tt = tic;
    videoID  = videos{vid};
    pathsSaveName = [saveName,videoID,'-actionpaths.mat'];   
    videoDetDirBase = [detresultpathBase,videoID,'/'];
    videoTopDirBase = [detresultpathTop,videoID,'/'];
    if ~exist(pathsSaveName,'file')
        fprintf('computing tubes for vide [%d out of %d] video ID = %s\n',vid,NumVideos, videoID);

        fprintf('Reading detection files searlially ');
        frames = readDetections(videoDetDirBase,videoTopDirBase);
        fprintf('\nDone reading detection files \n');
        fprintf('Gernrating action paths ...........\n');
        
        %% parllel loop over all action class and genrate paths for each class
        thpath = tic;
        allpaths = cell(1);
        for a=1:numActions
            allpaths{a} = genActionPaths(frames,a,nms_thresh,dopts.fuseiouth,dopts.fusiontype,iouth,costtype,gap);
        end
        timimngs(vid) = toc(thpath);
        %%
        fprintf('Completed linking \n');
        fprintf('results are being saved in::: %s\n',pathsSaveName);
        save(pathsSaveName,'allpaths');
        fprintf('All Done in %03d Seconds\n',round(toc(tt)));
    end
end

% save('ucf101timing.mat','numfs','timimngs')
disp('done computing action paths');
end

% ---------------------------------------------------------
% function to gather the detection box and nms them and pass it to linking script
function paths = genActionPaths(frames,a,nms_thresh,fuseiouth,fusiontype,iouth,costtype,gap)
% ---------------------------------------------------------
action_frames = struct();
for f=1:length(frames)

    baseBoxes = frames(f).baseBoxes;
    baseAllScores = frames(f).baseScores;
    topBoxes = frames(f).topBoxes;
    topAllScores = frames(f).topScores;
    meanScores = frames(f).meanScores;
    [boxes, allscores] = fuseboxes(baseBoxes,topBoxes,baseAllScores,topAllScores,meanScores,fuseiouth,fusiontype,a,nms_thresh);
    
    action_frames(f).allScores = allscores;
    action_frames(f).boxes = boxes(:,1:4);
    action_frames(f).scores = boxes(:,5);
end

paths = incremental_linking(action_frames,iouth,costtype,gap, gap);
end

% ---------------------------------------------------------
function [boxes,allscores] = fuseboxes(baseBoxes,topBoxes,baseAllScores,topAllScores,meanScores,fuseiouth,fusiontype,a,nms_thresh)
% ---------------------------------------------------------

if strcmp(fusiontype,'mean')
    [boxes,allscores] = dofilter(baseBoxes,meanScores,a,nms_thresh);
elseif strcmp(fusiontype,'nwsum-plus')
    [baseBoxes,baseAllScores] = dofilter(baseBoxes,baseAllScores,a,nms_thresh);
    [topBoxes,topAllScores] = dofilter(topBoxes,topAllScores,a,nms_thresh);
    [boxes,allscores] = boost_fusion(baseBoxes,topBoxes,baseAllScores,topAllScores,fuseiouth,a);
    pick = nms(boxes,nms_thresh);
    boxes = boxes(pick(1:min(10,length(pick))),:);
    allscores = allscores(pick(1:min(10,length(pick))),:);

else %% fusion type is cat // union-set fusion
    [baseBoxes,baseAllScores] = dofilter(baseBoxes,baseAllScores,a,nms_thresh);
    [topBoxes,topAllScores] = dofilter(topBoxes,topAllScores,a,nms_thresh);
    boxes = [baseBoxes;topBoxes];
    allscores = [baseAllScores;topAllScores];
    pick = nms(boxes,nms_thresh);
    boxes = boxes(pick(1:min(10,length(pick))),:);
    allscores = allscores(pick(1:min(10,length(pick))),:);
end

end


function [boxes,allscores] = dofilter(boxes, allscores,a,nms_thresh)
 scores = allscores(:,a);
 pick = scores>0.001;
 scores = scores(pick);
 boxes = boxes(pick,:);
 allscores = allscores(pick,:);
 [~,pick] = sort(scores,'descend');
 to_pick = min(50,size(pick,1));
 pick = pick(1:to_pick);
 scores = scores(pick);
 boxes = boxes(pick,:);
 allscores = allscores(pick,:);
 pick = nms([boxes scores], nms_thresh);
 pick = pick(1:min(10,length(pick)));
 boxes = [boxes(pick,:),scores(pick,:)];
 allscores = allscores(pick,:);
end

% ---------------------------------------------------------
function [sb,ss] = boost_fusion(sb, fb,ss,fs,fuseiouth,a) % bs - boxes_spatial bf-boxes_flow
% ---------------------------------------------------------

nb = size(sb,1); % num boxes
box_spatial = [sb(:,1:2) sb(:,3:4)-sb(:,1:2)+1];
box_flow =    [fb(:,1:2) fb(:,3:4)-fb(:,1:2)+1];
coveredboxes = [];

for i=1:nb
    ovlp = inters_union(box_spatial(i,:), box_flow); % ovlp has 1x5 or 5x1 dim
    if ~isempty(ovlp)
    [movlp, maxind] = max(ovlp);

    if movlp>=fuseiouth && isempty(ismember(coveredboxes,maxind))
        ms = ss(i,:) + fs(maxind,:)*movlp;
        ms = ms/sum(ms);
        sb(i,5) = ms(a);
        ss(i,:) = ms;
        coveredboxes = [coveredboxes;maxind];
    end
    end
end

nb = size(fb,1);

for i=1:nb
    if ~ismember(coveredboxes,i)
        sb = [sb;fb(i,:)];
        ss = [ss;fs(i,:)];
    end
end
end


function iou = inters_union(bounds1,bounds2)
% ------------------------------------------------------------------------
inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;
iou = inters./(union+0.001);
end

% -------------------------------------------------------------------------
function list = sortdirlist(dirname)
list = dir(dirname);
list = sort({list.name});
end

% -------------------------------------------------------------------------
function [videos] = getVideoNames(split_file)
% -------------------------------------------------------------------------
fprintf('Get both lis  %s\n',split_file);
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

function frames = readDetections(detectionDir,top_detectionDir )

detectionList = sortdirlist([detectionDir,'*.mat']);
frames = struct([]);
numframes = length(detectionList);
scores = 0;
loc = 0;
for f = 1 : numframes
    filename = [detectionDir,detectionList{f}];
    load(filename); % load loc and scores variable
    loc = [loc(:,1)*320, loc(:,2)*240, loc(:,3)*320, loc(:,4)*240];
    loc(loc(:,1)<0,1) = 0;
    loc(loc(:,2)<0,2) = 0;
    loc(loc(:,3)>319,3) = 319;
    loc(loc(:,4)>239,4) = 239;
    loc = loc + 1;
    frames(f).baseBoxes = loc;
    frames(f).baseScores = [scores(:,2:end),scores(:,1)];
    
    filename = [top_detectionDir,detectionList{f}];
    load(filename); % load loc and scores variable
    loc = [loc(:,1)*320, loc(:,2)*240, loc(:,3)*320, loc(:,4)*240];
    loc(loc(:,1)<0,1) = 0;
    loc(loc(:,2)<0,2) = 0;
    loc(loc(:,3)>319,3) = 319;
    loc(loc(:,4)>239,4) = 239;
    loc = loc + 1;
    frames(f).topBoxes = loc;
    frames(f).topScores = [scores(:,2:end),scores(:,1)];
    frames(f).meanScores = (frames(f).topScores + frames(f).baseScores)/2.0;
end

end


