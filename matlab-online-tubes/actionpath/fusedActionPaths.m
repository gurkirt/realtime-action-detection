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
fuseiouth = dopts.fuseiouth;
fusiontype = dopts.fusiontype;
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
    frames = readDetections(videoDetDirBase,videoTopDirBase);
    fprintf('\nDone reading detection files \n');
    fprintf('Gernrating action paths ...........\n');
    %% parllel loop over all action class and genrate paths for each class
    threshold = [0.01,0.5,0.9];
    for iiii = 2:2
        dis_thres = threshold(iiii);
        
    my_live_paths = cell(1); %% Stores live paths
    my_dead_paths = cell(1); %% Store the paths that has been terminated
    
    for i = 1:24
        my_live_paths{i} = struct();
        my_dead_paths{i} = struct();
        my_dead_paths{i}.dp_count = 0;
    end
    
    action_frames = struct();
    for f=1:length(frames)
        for a=1:numActions
            baseBoxes = frames(f).baseBoxes;
            baseAllScores = frames(f).baseScores;
            topBoxes = frames(f).topBoxes;
            topAllScores = frames(f).topScores;
            meanScores = frames(f).meanScores;
            [boxes, allscores] = fuseboxes(baseBoxes,topBoxes,baseAllScores,topAllScores,meanScores,fuseiouth,fusiontype,a,nms_thresh);

            action_frames(f).allScores = allscores;
            action_frames(f).boxes = boxes(:,1:4);
            action_frames(f).scores = boxes(:,5);
    
            [my_live_paths{a}, my_dead_paths{a}] = incremental_linking(f, action_frames, iouth, costtype, gap,...
                my_live_paths{a}, my_dead_paths{a},a);
        end
        
        strr =  strcat('/home/zhujiagang/realtime-action-detection/ucf24/rgb-images/', videoID, '/', num2str(f, '%05d'), '.jpg');
        img = imread(strr);
        
        dis_boxes = [];
        for a=1:numActions
            %size(my_live_paths{a}, 2)
            if size(my_live_paths{a}, 2) > 0
                for ii = 1:size(my_live_paths{a}, 2)
                    if isfield(my_live_paths{a}(ii),'scores')
                        if my_live_paths{a}(ii).foundAT(end) == f
                            if my_live_paths{a}(ii).scores(end) > dis_thres
                                count = my_live_paths{a}(ii).count;
                                dis_boxes = [dis_boxes;my_live_paths{a}(ii).boxes(count,:), my_live_paths{a}(ii).scores(end), a];
                                pt = round(my_live_paths{a}(ii).boxes(count,1:2));
                                wSize = round(my_live_paths{a}(ii).boxes(count,3:4) - my_live_paths{a}(ii).boxes(count,1:2));
                                
                                %% adding boxes to images
                                img = drawRect(img, pt, wSize);                                
                            end
                        end
                    end
                end
            end
        end
        %% display images, scores and boxes    
        if size(dis_boxes,1)>0
            strcell=cell(size(dis_boxes,1),1);        
            for iii=1:size(dis_boxes,1)
                strcell(iii) = {strcat(actions{dis_boxes(iii,6)}, ': ', num2str(dis_boxes(iii,5),3))};
            end
            RGB = insertText(img, double(dis_boxes(:,1:2)), strcell);
            imshow(RGB)
            str_save_dir =  strcat('/home/zhujiagang/realtime-action-detection/online_save/',videoID,'_', num2str(dis_thres));
            if ~exist(str_save_dir) 
                mkdir(str_save_dir)
            end
            str_save =  strcat(str_save_dir, '/', num2str(f, '%05d'), '.jpg');
            imwrite(RGB, str_save);
        end
    end

        fprintf('All Done in %03d Seconds\n',round(toc));
    end
    
    disp('done computing action paths');
    %%%%%%%%%%%%%%
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


function [ dest ] = drawRect( src, pt, wSize,  lineSize, color )
flag = 2;

if nargin < 5
    color = [255 255 0];
end

if nargin < 4
    lineSize = 1;
end

if nargin < 3
    disp('inenough parameters')
    return;
end

[yA, xA, z] = size(src);
x1 = pt(1);
y1 = pt(2);

wx = wSize(1);
wy = wSize(2);

if x1>xA 
   x1 = xA;
end
if x1<1 
   x1 = 1;
end

if y1>yA
   y1 = yA;
end
if y1<1
   y1 = 1;
end

if (x1+wx)>xA
    wx = xA - x1;
end
if (y1+wy)>yA
    wy = yA - y1;
end

if (x1+wx)<1
    wx = 1;
end
if (y1+wy)<1
    wy = 1;
end

if 1==z
    dest(:, : ,1) = src;
    dest(:, : ,2) = src;
    dest(:, : ,3) = src;
else
    dest = src;
end


for c = 1 : 3                
    for dl = 1 : lineSize    
        d = dl - 1;
        if  1==flag  
            dest(  y1-d ,            x1:(x1+wx) ,  c  ) =  color(c); 
            dest(  y1+wy+d ,     x1:(x1+wx) , c  ) =  color(c); 
            dest(  y1:(y1+wy) ,   x1-d ,           c  ) =  color(c); 
            dest(  y1:(y1+wy) ,   x1+wx+d ,    c  ) =  color(c); 
        elseif 2==flag 
            dest(  y1-d ,            (x1-d):(x1+wx+d) ,  c  ) =  color(c); 
            dest(  y1+wy+d ,    (x1-d):(x1+wx+d) ,  c  ) =  color(c); 
            dest(  (y1-d):(y1+wy+d) ,   x1-d ,           c  ) =  color(c); 
            dest(  (y1-d):(y1+wy+d) ,   x1+wx+d ,    c  ) =  color(c); 
        end
    end
end 

end 
