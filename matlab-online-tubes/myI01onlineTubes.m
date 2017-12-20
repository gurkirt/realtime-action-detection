% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
%% This is main script to build tubes and evaluate them %%

function myI01onlineTubes()
close all
data_root = '/home/zhujiagang/realtime-action-detection';
save_root = '/home/zhujiagang/realtime-action-detection/save';
iteration_num_rgb = [120000]; % you can also evaluate on multiple iertations
iteration_num_flow = [120000]; % you can also evaluate on multiple iertations

% add subfolder to matlab paths
addpath(genpath('gentube/'));
addpath(genpath('actionpath/'));
addpath(genpath('eval/'));
addpath(genpath('utils/'));
model_type = 'CONV';

completeList = {...
    {'ucf24','01',{'rgb'},iteration_num_rgb,{'score'}},...
    {'ucf24','01',{'brox'},iteration_num_flow,{'score'}}...
    {'ucf24','01',{'fastOF'},iteration_num_flow,{'score'}}...
    };

alldopts = cell(2,1);
count = 1;
gap=3;

for setind = 1:length(completeList)
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

%% For each option type build tubes and evaluate them
for index = 1:count-1
    opts = alldopts{index};
    if exist(opts.detDir,'dir')
        fprintf('Video List %02d :: %s\nAnnotFile :: %s\nImage  Dir :: %s\nDetection Dir:: %s\nActionpath Dir:: %s\nTube Dir:: %s\n',...
            index, opts.vidList, opts.annotFile, opts.imgDir, opts.detDir, opts.actPathDir, opts.tubeDir);
        %% online bbx and prediction scores display given frame level detections
        actionPaths(opts);
    end
end



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

annot = load(dopts.annotFile);
annot = annot.annot;
testvideos = getVideoNames(dopts.vidList);

for alpha = 3 
    fprintf('alpha %03d ', alpha);
    % read action paths
    actionpaths = readALLactionPaths(dopts.vidList,dopts.actPathDir,1);
    %% perform temporal trimming
    smoothedtubes = parActionPathSmoother(actionpaths,alpha*ones(numActions,1),numActions);
    fprintf('\n');
end

results(counter:end,:) = [];
result = cell(2,1);
result{2} = class_aps;
result{1} = results;
results = result;


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
