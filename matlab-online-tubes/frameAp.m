% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

%% This is main script to compute frame mean AP %%
%% this code is very new so hasn't been tested a lot
% Input: Detection directory; annotation file path; split file path
% Output: computes frame AP for all the detection directories
% It should produce results almost identical to test_ucf24.py

function frameAp()

addpath(genpath('eval/'));
addpath(genpath('utils/'));
addpath(genpath('actionpath/'));
data_root = '/home/zhujiagang/realtime-action-detection';
save_root = '/home/zhujiagang/realtime-action-detection/save';
iou_th = 0.5;
model_type = 'CONV';
dataset = 'ucf24';
list_id = '01';
split_file = sprintf('%s/%s/splitfiles/t%s.txt',data_root,dataset,list_id);
annotfile = sprintf('%s/%s/splitfiles/annots.mat',data_root,dataset);
annot = load(annotfile);
annot = annot.annot;
testlist = getVideoNames(split_file);
num_vid = length(testlist);
num_actions = 24;

logfile = fopen('frameAP.log','w'); % open log file

imgType = 'rgb'; iteration_num = 120000;
det_dirs1 = sprintf('%s/%s/detections/%s-%s-%s-%06d/',save_root,dataset,model_type,imgType,list_id,iteration_num);
imgType = 'brox'; iteration_num = 120000;
det_dirs2 = sprintf('%s/%s/detections/%s-%s-%s-%06d/',save_root,dataset,model_type,imgType,list_id,iteration_num);
imgType = 'fastOF'; iteration_num = 120000;
det_dirs3 = sprintf('%s/%s/detections/%s-%s-%s-%06d/',save_root,dataset,model_type,imgType,list_id,iteration_num);

combinations = {{det_dirs1},{det_dirs2},{det_dirs3},...
    {det_dirs1,det_dirs3,'boost'},{det_dirs1,det_dirs2,'boost'},...
    {det_dirs1,det_dirs3,'cat'},{det_dirs1,det_dirs2,'cat'},...
    {det_dirs1,det_dirs3,'mean'},{det_dirs1,det_dirs2,'mean'}};

for c=1:length(combinations)
    comb = combinations{c};
    line = comb{1};
    if length(comb)>1
        fusion_type = comb{3};
        line = [line,' ',comb{2},' \n\n fusion type: ',fusion_type,'\n\n'];
        
    else
        fusion_type = 'none';
    end
    
    line = sprintf('Evaluation for %s\n',line);
    fprintf('%s',line)
    fprintf(logfile,'%s',line);
    AP = zeros(num_actions,1);
    cc = zeros(num_actions,1);
    for a=1:num_actions
        allscore{a} = zeros(24*20*160000,2,'single');
    end
    
    total_num_gt_boxes = zeros(num_actions,1);
    annotNames = {annot.name};
    
    for vid=1:num_vid
        video_name = testlist{vid};
        [~,gtVidInd] = find(strcmp(annotNames, testlist{vid}));
        gt_tubes = annot(gtVidInd).tubes;
        numf = annot(gtVidInd).num_imgs;
        num_gt_tubes = length(gt_tubes);
        if mod(vid,5) == 0
            fprintf('Done procesing %d videos out of %d %s\n', vid, num_vid, video_name)
        end
        for nf = 1:numf
            gt_boxes = get_gt_boxes(gt_tubes,nf);
            dt_boxes = get_dt_boxes(comb, video_name, nf, num_actions, fusion_type);
            num_gt_boxes = size(gt_boxes,1);
            for g = 1:num_gt_boxes
                total_num_gt_boxes(gt_boxes(g,5)) = total_num_gt_boxes(gt_boxes(g,5)) + 1;
            end
            covered_gt_boxes = zeros(num_gt_boxes,1);
            for d = 1 : size(dt_boxes,1)
                dt_score = dt_boxes(d,5);
                dt_label = dt_boxes(d,6);
                cc(dt_label) = cc(dt_label) + 1;
                ioumax=-inf; maxgtind=0;
                if num_gt_boxes>0  && any(gt_boxes(:,5) == dt_label)
                    for g = 1:num_gt_boxes
                        if ~covered_gt_boxes(g) && any(dt_label == gt_boxes(:,5))
                            iou = compute_spatial_iou(gt_boxes(g,1:4), dt_boxes(d,1:4));
                            if iou>ioumax
                                ioumax=iou;
                                maxgtind=g;
                            end
                        end
                    end
                end
                
                if ioumax>=iou_th
                    covered_gt_boxes(maxgtind) = 1;
                    allscore{dt_label}(cc(dt_label),:) = [dt_score,1]; % tp detection
                else
                    allscore{dt_label}(cc(dt_label),:) = [dt_score,0]; % fp detection
                end
                
            end
            
        end
    end
    % Sort scores and then reorder tp fp labels in result precision and recall for each action
    for a=1:num_actions
        allscore{a} = allscore{a}(1:cc(a),:);
        scores = allscore{a}(:,1);
        labels = allscore{a}(:,2);
        [~, si] = sort(scores,'descend');
        %     scores = scores(si);
        labels = labels(si);
        fp=cumsum(labels==0);
        tp=cumsum(labels==1);
        recall=tp/total_num_gt_boxes(a);
        precision=tp./(fp+tp);
        AP(a) = xVOCap(recall,precision);
        line = sprintf('Action %02d AP = %0.5f \n', a, AP(a));
        fprintf('%s',line);
        fprintf(logfile,'%s',line);
    end
    
    AP(isnan(AP)) = 0;
    mAP  = mean(AP);
    line = sprintf('\nMean AP::=> %.5f\n\n',mAP);
    fprintf('%s',line);
    fprintf(logfile,'%s',line);
end
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

function gt_boxes = get_gt_boxes(gt_tubes,nf)
gt_boxes = [];
gt_tubes;
for t = 1:length(gt_tubes)
    if nf >= gt_tubes(t).sf && nf <= gt_tubes(t).ef
        b_ind = nf - gt_tubes(t).sf + 1;
        box = [gt_tubes(t).boxes(b_ind,:), gt_tubes(t).class];
        gt_boxes = [gt_boxes;box];
    end
end
end

function dt_boxes = get_dt_boxes(detection_dir, video_name, nf, num_actions, fusion_type)
dt_boxes = [];
%% apply nms per class
[boxes,scores] = read_detections(detection_dir, video_name, nf);
for a = 1 : num_actions
    cls_boxes = get_cls_detection(boxes,scores,a,fusion_type);
    dt_boxes = [dt_boxes; cls_boxes];
end
end

function cls_boxes = get_cls_detection(boxes,scores,a,fusion_type)

if strcmp(fusion_type,'none')
    cls_boxes = dofilter(boxes(1).b,scores(1).s,a);
elseif strcmp(fusion_type,'mean')
    cls_boxes = dofilter(boxes(1).b,(scores(1).s+scores(2).s)/2.0,a);
elseif strcmp(fusion_type,'cat')
    cls_boxes_base = dofilter(boxes(1).b,scores(1).s,a);
    cls_boxes_top = dofilter(boxes(2).b,scores(2).s,a);
    all_boxes = [cls_boxes_base;cls_boxes_top];
    pick = nms(all_boxes(:,1:5),0.45);
    cls_boxes = all_boxes(pick,:);
elseif strcmp(fusion_type,'boost')
    cls_boxes_base = dofilter(boxes(1).b,scores(1).s,a);
    cls_boxes_top = dofilter(boxes(2).b,scores(2).s,a);
    all_boxes = boost_boxes(cls_boxes_base,cls_boxes_top);
    pick = nms(all_boxes(:,1:5),0.45);
    cls_boxes = all_boxes(pick,:);
else
    error('Spacify correct fusion technique');
end

end

function cls_boxes_base = boost_boxes(cls_boxes_base,cls_boxes_top)

box_spatial = [cls_boxes_base(:,1:2) cls_boxes_base(:,3:4)-cls_boxes_base(:,1:2)+1];
box_flow =    [cls_boxes_top(:,1:2) cls_boxes_top(:,3:4)-cls_boxes_top(:,1:2)+1];
coveredboxes = [];
nb = size(cls_boxes_base,1); % num boxes
for i=1:nb
    ovlp = inters_union(box_spatial(i,:), box_flow); % ovlp has 1x5 or 5x1 dim
    if ~isempty(ovlp)
        [movlp, maxind] = max(ovlp);
        if movlp>=0.3 && isempty(ismember(coveredboxes,maxind))
            cls_boxes_base(i,5) = cls_boxes_base(i,5) + cls_boxes_top(maxind,5)*movlp;
            coveredboxes = [coveredboxes;maxind];
        end
    end
end

nb = size(cls_boxes_top,1);
for i=1:nb
    if ~ismember(coveredboxes,i)
        cls_boxes_base = [cls_boxes_base; cls_boxes_top(i,:)];
    end
end

end

function [bxs, sc] = read_detections(detection_dir, video_name, nf)
detection_dir1 = detection_dir{1};
det_file = sprintf('%s%s/%05d.mat', detection_dir1, video_name, nf);
load(det_file); % loads loc and scores variable
boxes = [loc(:,1)*320, loc(:,2)*240, loc(:,3)*320, loc(:,4)*240] + 1;
boxes(boxes(:,1)<1,1) = 1;   boxes(boxes(:,2)<1,2) = 1;
boxes(boxes(:,3)>320,3) = 320;  boxes(boxes(:,4)>240,4) = 240;
scores = [scores(:,2:end),scores(:,1)];
bxs = struct();
sc = struct();
bxs(1).b = boxes;
sc(1).s = scores;
if length(detection_dir)>1
    detection_dir1 = detection_dir{2};
    det_file = sprintf('%s%s/%05d.mat', detection_dir1, video_name, nf);
    load(det_file); % loads loc and scores variable
    boxes = [loc(:,1)*320, loc(:,2)*240, loc(:,3)*320, loc(:,4)*240] + 1;
    boxes(boxes(:,1)<1,1) = 1;   boxes(boxes(:,2)<1,2) = 1;
    boxes(boxes(:,3)>320,3) = 320;  boxes(boxes(:,4)>240,4) = 240;
    scores = [scores(:,2:end),scores(:,1)];
    bxs(2).b = boxes;
    sc(2).s = scores;
end

end


function boxes = dofilter(boxes,scores,a)
scores = scores(:,a);
pick = scores>0.01;
scores = scores(pick);
boxes = boxes(pick,:);
[~,pick] = sort(scores,'descend');
to_pick = min(50,size(pick,1));
pick = pick(1:to_pick);
scores = scores(pick);
boxes = boxes(pick,:);
pick = nms([boxes scores],0.45);
pick = pick(1:min(20,length(pick)));
boxes = boxes(pick,:);
scores = scores(pick);
cls = scores*0 + a;
boxes = [boxes,scores, cls];
end

function iou = inters_union(bounds1,bounds2)
% ------------------------------------------------------------------------
inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;
iou = inters./(union+0.001);
end


function iou = compute_spatial_iou(gt_box, dt_box)
dt_box = [dt_box(1:2), dt_box(3:4)-dt_box(1:2)+1];
inter = rectint(gt_box,dt_box);
ar1 = gt_box(3)*gt_box(4);
ar2 = dt_box(3)*dt_box(4);
union = ar1 + ar2 - inter;
iou = inter/union;
end