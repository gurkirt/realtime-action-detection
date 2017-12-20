% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
% Input: smoothed tubes
% Output: filtered out tubes with proper scoring

function xmld = convert2eval(final_tubes,min_num_frames,kthresh,topk,vids)

xmld = struct([]);
v= 1;

for vv = 1 :  length(vids)
    action_indexes = find(strcmp(final_tubes.video_id,vids{vv}));
    videoName = vids{vv};
    xmld(v).videoName = videoName;
    actionscore = final_tubes.dpActionScore(action_indexes);
    path_scores = final_tubes.path_scores(1,action_indexes);
    
    ts = final_tubes.ts(action_indexes);
    starts = final_tubes.starts(action_indexes);
    te = final_tubes.te(action_indexes);
    act_nr = 1;
     
    for a = 1 : length(ts)
        act_ts = ts(a);
        act_te = te(a);
%         act_dp_score = actionscore(a); %% only useful on JHMDB
        act_path_scores = cell2mat(path_scores(a));
        
        %-----------------------------------------------------------
        act_scores = sort(act_path_scores(act_ts:act_te),'descend');   
        %save('test.mat', 'act_scores'); pause;
        
        topk_mean = mean(act_scores(1:min(topk,length(act_scores))));        
        
        bxs = final_tubes.path_boxes{action_indexes(a)}(act_ts:act_te,:);
        
        bxs = [bxs(:,1:2), bxs(:,3:4)-bxs(:,1:2)];
        
        label = final_tubes.label(action_indexes(a));
        
        if topk_mean > kthresh(label) && (act_te-act_ts) > min_num_frames 
            xmld(v).score(act_nr) = topk_mean;
            xmld(v).nr(act_nr) = act_nr;
            xmld(v).class(act_nr) = label;
            xmld(v).framenr(act_nr).fnr = (act_ts:act_te) + starts(a)-1;
            xmld(v).boxes(act_nr).bxs = bxs;
            act_nr = act_nr+1;
        end
    end
    v = v + 1;

end
