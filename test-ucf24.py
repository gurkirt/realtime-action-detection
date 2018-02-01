"""
    Copyright (c) 2017, Gurkirt Singh

    This code and is available
    under the terms of MIT License provided in LICENSE.
    Please retain this notice and LICENSE if you use
    this file (or any portion of it) in your project.
    ---------------------------------------------------------
"""

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import AnnotationTransform, UCF24Detection, BaseTransform, CLASSES, detection_collate, v2
from ssd import build_ssd
import torch.utils.data as data
from layers.box_utils import decode, nms
from utils.evaluation import evaluate_detections
import os, time
import argparse
import numpy as np
import pickle
import scipy.io as sio # to save detection as mat files
cfg = v2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb can take flow as well')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--eval_iter', default='120000,', type=str, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--data_root', default='/mnt/mars-fast/datasets/', help='Location of VOC root directory')
parser.add_argument('--save_root', default='/mnt/mars-gamma/datasets/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=20, type=int, help='topk for evaluation')

args = parser.parse_args()

if args.input_type != 'rgb':
    args.conf_thresh = 0.05

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_net(net, save_root, exp_name, input_type, dataset, iteration, num_classes, thresh=0.5 ):
    """ Test a SSD network on an Action image database. """

    val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                            shuffle=False, collate_fn=detection_collate, pin_memory=True)
    image_ids = dataset.ids
    save_ids = []
    val_step = 250
    num_images = len(dataset)
    video_list = dataset.video_list
    det_boxes = [[] for _ in range(len(CLASSES))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    num_batches = len(val_data_loader)
    det_file = save_root + 'cache/' + exp_name + '/detection-'+str(iteration).zfill(6)+'.pkl'
    print('Number of images ', len(dataset),' number of batchs', num_batches)
    frame_save_dir = save_root+'detections/CONV-'+input_type+'-'+args.listid+'-'+str(iteration).zfill(6)+'/'
    print('\n\n\nDetections will be store in ',frame_save_dir,'\n\n')
    for val_itr in range(len(val_data_loader)):
        if not batch_iterator:
            batch_iterator = iter(val_data_loader)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        images, targets, img_indexs = next(batch_iterator)
        batch_size = images.size(0)
        height, width = images.size(2), images.size(3)

        if args.cuda:
            images = Variable(images.cuda(), volatile=True)
        output = net(images)

        loc_data = output[0]
        conf_preds = output[1]
        prior_data = output[2]

        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            tf = time.perf_counter()
            print('Forward Time {:0.3f}'.format(tf - t1))
        for b in range(batch_size):
            gt = targets[b].numpy()
            gt[:, 0] *= width
            gt[:, 2] *= width
            gt[:, 1] *= height
            gt[:, 3] *= height
            gt_boxes.append(gt)
            decoded_boxes = decode(loc_data[b].data, prior_data.data, cfg['variance']).clone()
            conf_scores = net.softmax(conf_preds[b]).data.clone()
            index = img_indexs[b]
            annot_info = image_ids[index]

            frame_num = annot_info[1]; video_id = annot_info[0]; videoname = video_list[video_id]
            output_dir = frame_save_dir+videoname
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_file_name = output_dir+'/{:05d}.mat'.format(int(frame_num))
            save_ids.append(output_file_name)
            sio.savemat(output_file_name, mdict={'scores':conf_scores.cpu().numpy(),'loc':decoded_boxes.cpu().numpy()})

            for cl_ind in range(1, num_classes):
                scores = conf_scores[:, cl_ind].squeeze()
                c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                scores = scores[c_mask].squeeze()
                # print('scores size',scores.size())
                if scores.dim() == 0:
                    # print(len(''), ' dim ==0 ')
                    det_boxes[cl_ind - 1].append(np.asarray([]))
                    continue
                boxes = decoded_boxes.clone()
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes = boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                scores = scores[ids[:counts]].cpu().numpy()
                boxes = boxes[ids[:counts]].cpu().numpy()
                # print('boxes sahpe',boxes.shape)
                boxes[:, 0] *= width
                boxes[:, 2] *= width
                boxes[:, 1] *= height
                boxes[:, 3] *= height

                for ik in range(boxes.shape[0]):
                    boxes[ik, 0] = max(0, boxes[ik, 0])
                    boxes[ik, 2] = min(width, boxes[ik, 2])
                    boxes[ik, 1] = max(0, boxes[ik, 1])
                    boxes[ik, 3] = min(height, boxes[ik, 3])

                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                det_boxes[cl_ind - 1].append(cls_dets)

            count += 1
        if val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te - ts))
            torch.cuda.synchronize()
            ts = time.perf_counter()
        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('NMS stuff Time {:0.3f}'.format(te - tf))
    print('Evaluating detections for itration number ', iteration)

    #Save detection after NMS along with GT
    with open(det_file, 'wb') as f:
        pickle.dump([gt_boxes, det_boxes, save_ids], f, pickle.HIGHEST_PROTOCOL)

    return evaluate_detections(gt_boxes, det_boxes, CLASSES, iou_thresh=thresh)


def main():

    means = (104, 117, 123)  # only support voc now

    exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset, args.input_type,
                            args.batch_size, args.basenet[:-14], int(args.lr * 100000))

    args.save_root += args.dataset+'/'
    args.data_root += args.dataset+'/'
    args.listid = '01' ## would be usefull in JHMDB-21
    print('Exp name', exp_name, args.listid)
    for iteration in [int(itr) for itr in args.eval_iter.split(',')]:
        log_file = open(args.save_root + 'cache/' + exp_name + "/testing-{:d}.log".format(iteration), "w", 1)
        log_file.write(exp_name + '\n')
        trained_model_path = args.save_root + 'cache/' + exp_name + '/ssd300_ucf24_' + repr(iteration) + '.pth'
        log_file.write(trained_model_path+'\n')
        num_classes = len(CLASSES) + 1  #7 +1 background
        net = build_ssd(300, num_classes)  # initialize SSD
        net.load_state_dict(torch.load(trained_model_path))
        net.eval()
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        print('Finished loading model %d !' % iteration)
        # Load dataset
        dataset = UCF24Detection(args.data_root, 'test', BaseTransform(args.ssd_dim, means), AnnotationTransform(),
                                 input_type=args.input_type, full_test=True)
        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        mAP, ap_all, ap_strs = test_net(net, args.save_root, exp_name, args.input_type, dataset, iteration, num_classes)
        for ap_str in ap_strs:
            print(ap_str)
            log_file.write(ap_str + '\n')
        ptr_str = '\nMEANAP:::=>' + str(mAP) + '\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()

if __name__ == '__main__':
    main()
