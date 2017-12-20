
% ######################################################################################################################################################################################
% We are here talking about spatio-temporal detections, i.e. a set of ground-truth bounding boxes that
%  I will denote by g_t, with t between t_g^b and t_g^e (beginning and end time of the ground-truth)
% versus a detection which is also a set of bounding boxes, denoted by d_t, with t between t_d^e et t_d^e.
%
% a) temporal iou =  T_i / T_u
%  this is the intersection over union between the timing of the the tubes,
% ie mathematically T_i / T_u with
% the intersection T_i = max(0,   max(t_g^b,t_d^b)-min(t_d^e,t_g^e) )
% and the union T_u = min(t_g^b,t_d^b)-max(t_d^e,t_g^e)
%
% b) for each t between max(tgb,tdb)-min(tde,tge), we compute the IoU between g_t and d_t, and average them
%
% Multiplying (a) and (b) is the same as computed the average of the spatial iou over all frames in T_u of the two tubes, with a spatial iou of 0 for frames where only one box exists.
% c) as this is standard in detection problem, if there are multiple detections for the same groundtruth detection, the first one is counted as positive and the other ones as negatives
% ######################################################################################################################################################################################
%{
gt_fnr = 1xn doube
gt_bb = nx4 doubld - [x y w h]
dt_fnr = 1xm double
dt_bb = mx4 double - [x y w h]
%}
% -------------------------------------------------------------------------
function st_iou = compute_spatio_temporal_iou(gt_fnr, gt_bb, dt_fnr, dt_bb)
% -------------------------------------------------------------------------

% time gt begin
tgb = gt_fnr(1);
% time gt end
tge = gt_fnr(end);
%time dt begin
tdb = dt_fnr(1);
tde = dt_fnr(end);
% temporal intersection
T_i = double(max(0, min(tge,tde)-max(tgb,tdb)));

if T_i>0
    T_i = T_i +1;
    % temporal union
    T_u = double(max(tge,tde) - min(tgb,tdb)+1);
    %temporal IoU
    T_iou = T_i/T_u;
    % intersect frame numbers
    int_fnr = max(tgb,tdb):min(tge,tde);
    
    % find the ind of the intersected frames in the detected frames
    [~,int_find_dt] = ismember(int_fnr, dt_fnr);
    [~,int_find_gt] = ismember(int_fnr, gt_fnr);
    
    assert(length(int_find_dt)==length(int_find_gt));
    
    iou = zeros(length(int_find_dt),1);
    for i=1:length(int_find_dt)
        if int_find_gt(i)<1
%             fprintf('error ')
            pf = pf;
        else
            pf = i;
        end
        
        gt_bound = gt_bb(int_find_gt(pf),:);
        dt_bound = dt_bb(int_find_dt(pf),:)+1;
        
        % gt_bound = [gt_bound(:,1:2) gt_bound(:,3:4)-gt_bound(:,1:2)];
        % dt_bound = [dt_bound(:,1:2) dt_bound(:,3:4)-dt_bound(:,1:2)];
        iou(i) = inters_union(double(gt_bound),double(dt_bound));
    end
    % finalspatio-temporal IoU threshold
    st_iou = T_iou*mean(iou);
else
    st_iou =0;
end
% % iou_thresh = 0.2,...,0.6 % 'Learing to track paper' takes 0.2 for UCF101 and 0.5 for JHMDB
% if delta >= iou_thresh
%     % consider this tube as valid detection
% end

end

% -------------------------------------------------------------------------
function iou = inters_union(bounds1,bounds2)
% -------------------------------------------------------------------------

inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;

iou = inters./(union+eps);

end
