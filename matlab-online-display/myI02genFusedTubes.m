
function myI02genFusedTubes()

data_root = '/home/zhujiagang/realtime-action-detection';
save_root = '/home/zhujiagang/realtime-action-detection/save';
iteration_num_rgb = 90000; % you can also evaluate on multiple iertations
iteration_num_flow = 120000; % you can also evaluate on multiple iertations

addpath(genpath('actionpath/'));
addpath(genpath('gentube/'));
addpath(genpath('eval/'));
addpath(genpath('utils/'));

completeList = {...
    {'ucf24','01',{'rgb','brox'},[90000,120000],{'cat','nwsum-plus','mean'}, 0.25},...
    {'ucf24','01',{'rgb','brox'},[120000,120000],{'cat','nwsum-plus','mean'}, 0.25},...
    {'ucf24','01',{'rgb','fastOF'},[90000,120000],{'cat','nwsum-plus','mean'}, 0.25},...
    {'ucf24','01',{'rgb','fastOF'},[120000,120000],{'cat','nwsum-plus','mean'}, 0.25},...
    };
model_type = 'CONV';
costtype = 'score';
iouthresh = 0.1;
gap = 3;
alldopts = cell(2,1);
count = 1;
for setind = [2,4] %1:length(completeList)
    [dataset,listid,imtypes,iteration_nums,fusiontypes,fuseiouths] = enumurateList(completeList{setind});
    for ff =1:length(fusiontypes)
        fusiontype = fusiontypes{ff};
        if strcmp(fusiontype,'cat') || strcmp(fusiontype,'mean')
            tempfuseiouths = 0;
        else
            tempfuseiouths = fuseiouths;
        end
        for fuseiouth = tempfuseiouths
            for iouWeight = 1
                dopts = initDatasetOptsFused(data_root,save_root,dataset,imtypes,model_type, ...
                            listid,iteration_nums,iouthresh,costtype,gap,fusiontype,fuseiouth);
                if exist(dopts.basedetDir,'dir') && exist(dopts.topdetDir,'dir')
                    alldopts{count} = dopts;
                    count = count+1;
                end
            end
        end
    end
end

fprintf('\n\n\n\n %d \n\n\n\n',count)

% sets = {1:12,13:24,25:36,49:64};
% parpool('local',16); %length(set));

for setid = 1
    for index = 1:count-1
        opts = alldopts{index};
        if exist(opts.basedetDir,'dir') && exist(opts.topdetDir,'dir')
            fprintf('Video List :: %s\n \nDetection basedetDir:: %s\nActionpath Dir:: %s\nTube Dir:: %s\n',...
                opts.vidList,opts.basedetDir,opts.actPathDir,opts.tubeDir);
            
            %% online bbx and prediction scores display given fused frame level detections
            fusedActionPaths(opts);
        end
    end
end


function [dataset,listnum,imtypes,weights,fusiontypes,fuseiouths] = enumurateList(sublist)

dataset = sublist{1}; listnum = sublist{2}; imtypes = sublist{3};
weights = sublist{4};
fusiontypes = sublist{5};
fuseiouths =  sublist{6};

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
    for  alpha = 3 
        fprintf('alpha %03d ',alpha);
        tubesSaveName = sprintf('%stubes-alpha%04d.mat',dopts.tubeDir,uint16(alpha*100));
        if ~exist(tubesSaveName,'file')
            % read action paths
            actionpaths = readALLactionPaths(dopts.vidList,dopts.actPathDir,1);
            %% perform temporal trimming
            smoothedtubes = parActionPathSmoother(actionpaths,alpha*ones(numActions,1),numActions);
            save(tubesSaveName,'smoothedtubes','-v7.3');
        else
            load(tubesSaveName)
        end
        
        min_num_frames = 8;    kthresh = 0.0;     topk = 40;
        % strip off uncessary parts and remove very small actions less than
        % 8 frames; not really necessary but for speed at eval time
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