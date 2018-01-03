% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
%% This is main script to build tubes and evaluate them %%

function I01onlineTubes()

data_root = '/mnt/sun-gamma/datasets';
save_root = '/mnt/sun-gamma/datasets';
iteration_nums = [70000,120000,50000,90000]; % you can also evaluate on multiple iterations

% add subfolder to matlab paths
addpath(genpath('gentube/'));
addpath(genpath('actionpath/'));
addpath(genpath('eval/'));
addpath(genpath('utils/'));
model_type = 'CONV';

completeList = {...
    {'ucf24','01', {'rgb'}, iteration_nums,{'score'}},...
    {'ucf24','01', {'brox'}, iteration_nums,{'score'}}...
    {'ucf24','01', {'fastOF'}, iteration_nums,{'score'}}...
    };

alldopts = cell(2,1);
count = 1;
gap=3;

for setind = 1 %:length(completeList)
    [dataset, listid, imtypes, iteration_nums, costTypes] = enumurateList(completeList{setind});
    for ct = 1:length(costTypes)
        costtype = costTypes{ct};
        for imtind = 1:length(imtypes)
            imgType = imtypes{imtind};
            for iteration = iteration_nums
                for iouthresh=0.1
                    %% generate directory sturcture based on the options
                    dopts = initDatasetOpts(data_root,save_root,dataset,imgType,model_type,listid,iteration,iouthresh,costtype, gap);
                    if exist(dopts.detDir,'dir')
                        alldopts{count} = dopts;
                        count = count+1;
                    end
                end
            end
        end
    end
end

results = cell(2,1);

%% For each option type build tubes and evaluate them
for index = 1:count-1
    opts = alldopts{index};
    if exist(opts.detDir,'dir')
        fprintf('Video List %02d :: %s\nAnnotFile :: %s\nImage  Dir :: %s\nDetection Dir:: %s\nActionpath Dir:: %s\nTube Dir:: %s\n',...
            index, opts.vidList, opts.annotFile, opts.imgDir, opts.detDir, opts.actPathDir, opts.tubeDir);
        %% Build action paths given frame level detections
        actionPaths(opts);
        %% Perform temproal labelling and evaluate; results saved in results cell
        result_cell = gettubes(opts);
        results{index,1} = result_cell;
        results{index,2} = opts;
        rm = result_cell{1};
        rm = rm(rm(:,2) == 5,:);
        fprintf('\nmAP@0.2:%0.4f mAP@0.5:%0.4f mAP@0.75:%0.4f AVGmAP:%0.4f clsAcc:%0.4f\n\n',...
                    rm(1,5),rm(2,5),rm(7,5),mean(rm(2:end,5)),rm(1,6));
    end
end

%% save results
save_dir = [save_root,'/results/'];
if ~isdir(save_dir)
    mkdir(save_dir)
end
save_dir
save([save_dir,'online_tubes_results_CONV.mat'],'results')

%% Function to enumrate options
function [dataset,listnum,imtypes,weights,costTypes] = enumurateList(sublist)
dataset = sublist{1}; listnum = sublist{2}; imtypes = sublist{3};
weights = sublist{4};costTypes = sublist{5};

%% Facade function for smoothing tubes and evaluating them
function results = gettubes(dopts)

numActions = length(dopts.actions);
results = zeros(300,6);
counter=1;
class_aps = cell(2,1);
% save file name to save result for eah option type
saveName = sprintf('%stubes-results.mat',dopts.tubeDir);
if ~exist(saveName,'file')
    
    annot = load(dopts.annotFile);
    annot = annot.annot;
    testvideos = getVideoNames(dopts.vidList);
    actionpaths = readALLactionPaths(dopts.vidList,dopts.actPathDir,1);
    for  alpha = [3, 5]
        fprintf('alpha %03d ',alpha);
        tubesSaveName = sprintf('%stubes-alpha%04d.mat',dopts.tubeDir,uint16(alpha*100));
        if ~exist(tubesSaveName,'file')
            % read action paths
            %% perform temporal trimming
            smoothedtubes = PARactionPathSmoother(actionpaths,alpha*ones(numActions,1),numActions);
            save(tubesSaveName,'smoothedtubes','-v7.3');
        else
            load(tubesSaveName)
        end
        
        min_num_frames = 8;    kthresh = 0.0;     topk = 40;
        xmldata = convert2eval(smoothedtubes, min_num_frames, kthresh*ones(numActions,1), topk,testvideos);
        
        %% Do the evaluation
        for iou_th =[0.2,[0.5:0.05:0.95]]
            [tmAP,tmIoU,tacc,AP] = get_PR_curve(annot, xmldata, testvideos, dopts.actions, iou_th);
            % pritn outs iou_threshold, meanAp, sm, classifcation accuracy
            fprintf('%.2f %0.3f %0.3f N ',iou_th,tmAP, tacc);
            results(counter,:) = [iou_th,alpha,alpha,tmIoU,tmAP,tacc];
            class_aps{counter} = AP;
            counter = counter+1;
        end
        fprintf('\n');
    end



    results(counter:end,:) = [];
    result = cell(2,1);
    result{2} = class_aps;
    result{1} = results;
    results = result;
    fprintf('results saved in %s\n',saveName);
    save(saveName,'results');
else
    load(saveName)
end

function videos = getVideoNames(split_file)
% -------------------------------------------------------------------------
fid = fopen(split_file,'r');
data = textscan(fid, '%s');
videos  = cell(1);
count = 0;
for i=1:length(data{1})
    filename = cell2mat(data{1}(i,1));
    count = count +1;
    videos{count} = filename;
end
