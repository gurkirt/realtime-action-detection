
""" Evaluation code based on VOC protocol

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

Updated by Gurkirt Singh for ucf101-24 dataset

"""

import os
import numpy as np

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # print('voc_ap() - use_07_metric:=' + str(use_07_metric))
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_gt_of_cls(gt_boxes, cls):
    cls_gt_boxes = []
    for i in range(len(gt_boxes)):
        if gt_boxes[i,-1] == cls:
            cls_gt_boxes.append(gt_boxes[i, :-1])
    return np.asarray(cls_gt_boxes)


def compute_iou(cls_gt_boxes, box):
    ious = np.zeros(cls_gt_boxes.shape[0])

    for m in range(ious.shape[0]):
        gtbox = cls_gt_boxes[m]

        xmin = max(gtbox[0],box[0])
        ymin = max(gtbox[1], box[1])
        xmax = min(gtbox[2], box[2])
        ymax = min(gtbox[3], box[3])
        iw = np.maximum(xmax - xmin, 0.)
        ih = np.maximum(ymax - ymin, 0.)
        if iw>0 and ih>0:
            intsc = iw*ih
        else:
            intsc = 0.0
        # print (intsc)
        union = (gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1]) + (box[2] - box[0]) * (box[3] - box[1]) - intsc
        ious[m] = intsc/union

    return ious

def evaluate_detections(gt_boxes, det_boxes, CLASSES=[], iou_thresh=0.5):

    ap_strs = []
    num_frames = len(gt_boxes)
    print('Evaluating for ', num_frames, 'frames')
    ap_all = np.zeros(len(CLASSES), dtype=np.float32)
    for cls_ind, cls in enumerate(CLASSES): # loop over each class 'cls'
        scores = np.zeros(num_frames * 220)
        istp = np.zeros(num_frames * 220)
        det_count = 0
        num_postives = 0.0
        for nf in range(num_frames): # loop over each frame 'nf'
                # if len(gt_boxes[nf])>0 and len(det_boxes[cls_ind][nf]):
                frame_det_boxes = np.copy(det_boxes[cls_ind][nf]) # get frame detections for class cls in nf
                cls_gt_boxes = get_gt_of_cls(np.copy(gt_boxes[nf]), cls_ind) # get gt boxes for class cls in nf frame
                num_postives += cls_gt_boxes.shape[0]
                if frame_det_boxes.shape[0]>0: # check if there are dection for class cls in nf frame
                    argsort_scores = np.argsort(-frame_det_boxes[:,-1]) # sort in descending order
                    for i, k in enumerate(argsort_scores): # start from best scoring detection of cls to end
                        box = frame_det_boxes[k, :-1] # detection bounfing box
                        score = frame_det_boxes[k,-1] # detection score
                        ispositive = False # set ispostive to false every time
                        if cls_gt_boxes.shape[0]>0: # we can only find a postive detection
                            # if there is atleast one gt bounding for class cls is there in frame nf
                            iou = compute_iou(cls_gt_boxes, box) # compute IOU between remaining gt boxes
                            # and detection boxes
                            maxid = np.argmax(iou)  # get the max IOU window gt index
                            if iou[maxid] >= iou_thresh: # check is max IOU is greater than detection threshold
                                ispositive = True # if yes then this is ture positive detection
                                cls_gt_boxes = np.delete(cls_gt_boxes, maxid, 0) # remove assigned gt box
                        scores[det_count] = score # fill score array with score of current detection
                        if ispositive:
                            istp[det_count] = 1 # set current detection index (det_count)
                            #  to 1 if it is true postive example
                        det_count += 1
        if num_postives<1:
            num_postives =1
        scores = scores[:det_count]
        istp = istp[:det_count]
        argsort_scores = np.argsort(-scores) # sort in descending order
        istp = istp[argsort_scores] # reorder istp's on score sorting
        fp = np.cumsum(istp == 0) # get false positives
        tp = np.cumsum(istp == 1) # get  true positives
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        recall = tp / float(num_postives) # compute recall
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # compute precision
        cls_ap = voc_ap(recall, precision) # compute average precision using voc2007 metric
        ap_all[cls_ind] = cls_ap
        # print(cls_ind,CLASSES[cls_ind], cls_ap)
        ap_str = str(CLASSES[cls_ind]) + ' : ' + str(num_postives) + ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)

    # print ('mean ap ', np.mean(ap_all))
    return np.mean(ap_all), ap_all, ap_strs


def save_detection_framewise(det_boxes, image_ids, iteration):
    det_save_dir = '/mnt/mars-beta/gur-workspace/use-ssd-data/UCF101/detections/RGB-01-{:06d}/'.format(iteration)
    print('Saving detections to', det_save_dir)
    num_images = len(image_ids)
    for idx in range(num_images):
        img_id = image_ids[idx]
        save_path = det_save_dir+img_id[:-5]
        if not os.path.isdir(save_path):
            os.system('mkdir -p '+save_path)
        fid = open(det_save_dir+img_id+'.txt','w')
        for cls_ind in range(len(det_boxes)):
            frame_det_boxes = det_boxes[cls_ind][idx]
            for d in range(len(frame_det_boxes)):
                line = str(cls_ind+1)
                for k in range(5):
                    line += ' {:f}'.format(frame_det_boxes[d,k])
                line += '\n'
                fid.write(line)
        fid.close()

