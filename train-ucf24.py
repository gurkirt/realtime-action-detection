
""" Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Which was adopated by: Ellis Brown, Max deGroot
    https://github.com/amdegroot/ssd.pytorch

    Further:
    Updated by Gurkirt Singh for ucf101-24 dataset
    Licensed under The MIT License [see LICENSE for details]
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, UCF24Detection, AnnotationTransform, detection_collate, CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
from utils.evaluation import evaluate_detections
from layers.box_utils import decode, nms
from utils import  AverageMeter
from torch.optim.lr_scheduler import MultiStepLR

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb options are [rgb,brox,fastOF]')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--max_iter', default=150000, type=int, help='Number of training iterations')
parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--stepvalues', default='30000,60000,100000', type=str, help='iter numbers where learing rate to be dropped')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--vis_port', default=8097, type=int, help='Port for Visdom Server')
parser.add_argument('--data_root', default='/mnt/mars-fast/datasets/', help='Location of VOC root directory')
parser.add_argument('--save_root', default='/mnt/mars-gamma/datasets/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')

## Parse arguments
args = parser.parse_args()
## set random seeds
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)


torch.set_default_tensor_type('torch.FloatTensor')


def main():
    args.cfg = v2
    args.train_sets = 'train'
    args.means = (104, 117, 123)
    num_classes = len(CLASSES) + 1
    args.num_classes = num_classes
    args.stepvalues = [int(val) for val in args.stepvalues.split(',')]
    args.loss_reset_step = 30
    args.eval_step = 10000
    args.print_step = 10

    ## Define the experiment Name will used to same directory and ENV for visdom
    args.exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset,
                args.input_type, args.batch_size, args.basenet[:-14], int(args.lr*100000))

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    net = build_ssd(300, args.num_classes)

    if args.cuda:
        net = net.cuda()

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()


    print('Initializing weights for extra layers and HEADs...')
    # initialize newly added layers' weights with xavier method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    if args.input_type == 'fastOF':
        print('Download pretrained brox flow trained model weights and place them at:::=> ',args.data_root + 'ucf24/train_data/brox_wieghts.pth')
        pretrained_weights = args.data_root + 'ucf24/train_data/brox_wieghts.pth'
        print('Loading base network...')
        net.load_state_dict(torch.load(pretrained_weights))
    else:
        vgg_weights = torch.load(args.data_root +'ucf24/train_data/' + args.basenet)
        print('Loading base network...')
        net.vgg.load_state_dict(vgg_weights)

    args.data_root += args.dataset + '/'

    parameter_dict = dict(net.named_parameters()) # Get parmeter of network in dictionary format wtih name being key
    params = []

    #Set different learning rate to bias layers and set their weight_decay to 0
    for name, param in parameter_dict.items():
        if name.find('bias') > -1:
            print(name, 'layer parameters will be trained @ {}'.format(args.lr*2))
            params += [{'params': [param], 'lr': args.lr*2, 'weight_decay': 0}]
        else:
            print(name, 'layer parameters will be trained @ {}'.format(args.lr))
            params += [{'params':[param], 'lr': args.lr, 'weight_decay':args.weight_decay}]

    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    scheduler = MultiStepLR(optimizer, milestones=args.stepvalues, gamma=args.gamma)
    train(args, net, optimizer, criterion, scheduler)


def train(args, net, optimizer, criterion, scheduler):
    log_file = open(args.save_root+"training.log", "w", 1)
    log_file.write(args.exp_name+'\n')
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')
    log_file.write(str(net))
    net.train()

    # loss counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    print('Loading Dataset...')
    train_dataset = UCF24Detection(args.data_root, args.train_sets, SSDAugmentation(args.ssd_dim, args.means),
                                   AnnotationTransform(), input_type=args.input_type)
    val_dataset = UCF24Detection(args.data_root, 'test', BaseTransform(args.ssd_dim, args.means),
                                 AnnotationTransform(), input_type=args.input_type,
                                 full_test=False)
    epoch_size = len(train_dataset) // args.batch_size
    print('Training SSD on', train_dataset.name)

    if args.visdom:

        import visdom
        viz = visdom.Visdom()
        viz.port = args.vis_port
        viz.env = args.exp_name
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 6)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['REG', 'CLS', 'AVG', 'S-REG', ' S-CLS', ' S-AVG']
            )
        )
        # initialize visdom meanAP and class APs plot
        legends = ['meanAP']
        for cls in CLASSES:
            legends.append(cls)
        val_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,args.num_classes)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Mean AP',
                title='Current SSD Validation mean AP',
                legend=legends
            )
        )


    batch_iterator = None
    train_data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    val_data_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, collate_fn=detection_collate, pin_memory=True)
    itr_count = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iteration = 0
    while iteration <= args.max_iter:
        for i, (images, targets, img_indexs) in enumerate(train_data_loader):

            if iteration > args.max_iter:
                break
            iteration += 1
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            # forward
            out = net(images)
            # backprop
            optimizer.zero_grad()

            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            scheduler.step()
            loc_loss = loss_l.data[0]
            conf_loss = loss_c.data[0]
            # print('Loss data type ',type(loc_loss))
            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)
            losses.update((loc_loss + conf_loss)/2.0)


            if iteration % args.print_step == 0 and iteration>0:
                if args.visdom:
                    losses_list = [loc_losses.val, cls_losses.val, losses.val, loc_losses.avg, cls_losses.avg, losses.avg]
                    viz.line(X=torch.ones((1, 6)).cpu() * iteration,
                        Y=torch.from_numpy(np.asarray(losses_list)).unsqueeze(0).cpu(),
                        win=lot,
                        update='append')


                torch.cuda.synchronize()
                t1 = time.perf_counter()
                batch_time.update(t1 - t0)

                print_line = 'Itration {:06d}/{:06d} loc-loss {:.3f}({:.3f}) cls-loss {:.3f}({:.3f}) ' \
                             'average-loss {:.3f}({:.3f}) Timer {:0.3f}({:0.3f})'.format(
                              iteration, args.max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                              cls_losses.avg, losses.val, losses.avg, batch_time.val, batch_time.avg)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                log_file.write(print_line+'\n')
                print(print_line)

                # if args.visdom and args.send_images_to_visdom:
                #     random_batch_index = np.random.randint(images.size(0))
                #     viz.image(images.data[random_batch_index].cpu().numpy())
                itr_count += 1

                if itr_count % args.loss_reset_step == 0 and itr_count > 0:
                    loc_losses.reset()
                    cls_losses.reset()
                    losses.reset()
                    batch_time.reset()
                    print('Reset accumulators of ', args.exp_name,' at', itr_count*args.print_step)
                    itr_count = 0

            if (iteration % args.eval_step == 0 or iteration == 5000) and iteration>0:
                torch.cuda.synchronize()
                tvs = time.perf_counter()
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), args.save_root+'ssd300_ucf24_' +
                           repr(iteration) + '.pth')

                net.eval() # switch net to evaluation mode
                mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)

                for ap_str in ap_strs:
                    print(ap_str)
                    log_file.write(ap_str+'\n')
                ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
                print(ptr_str)
                log_file.write(ptr_str)

                if args.visdom:
                    aps = [mAP]
                    for ap in ap_all:
                        aps.append(ap)
                    viz.line(
                        X=torch.ones((1, args.num_classes)).cpu() * iteration,
                        Y=torch.from_numpy(np.asarray(aps)).unsqueeze(0).cpu(),
                        win=val_lot,
                        update='append'
                            )
                net.train() # Switch net back to training mode
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
                print(prt_str)
                log_file.write(ptr_str)

    log_file.close()


def validate(args, net, val_data_loader, val_dataset, iteration_num, iou_thresh=0.5):
    """Test a SSD network on an image database."""
    print('Validating at ', iteration_num)
    num_images = len(val_dataset)
    num_classes = args.num_classes

    det_boxes = [[] for _ in range(len(CLASSES))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    val_step = 100
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()

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
            print('Forward Time {:0.3f}'.format(tf-t1))
        for b in range(batch_size):
            gt = targets[b].numpy()
            gt[:,0] *= width
            gt[:,2] *= width
            gt[:,1] *= height
            gt[:,3] *= height
            gt_boxes.append(gt)
            decoded_boxes = decode(loc_data[b].data, prior_data.data, args.cfg['variance']).clone()
            conf_scores = net.softmax(conf_preds[b]).data.clone()

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
                boxes[:,0] *= width
                boxes[:,2] *= width
                boxes[:,1] *= height
                boxes[:,3] *= height

                for ik in range(boxes.shape[0]):
                    boxes[ik, 0] = max(0, boxes[ik, 0])
                    boxes[ik, 2] = min(width, boxes[ik, 2])
                    boxes[ik, 1] = max(0, boxes[ik, 1])
                    boxes[ik, 3] = min(height, boxes[ik, 3])

                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)

                det_boxes[cl_ind-1].append(cls_dets)
            count += 1
        if val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
            torch.cuda.synchronize()
            ts = time.perf_counter()
        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('NMS stuff Time {:0.3f}'.format(te - tf))
    print('Evaluating detections for itration number ', iteration_num)
    return evaluate_detections(gt_boxes, det_boxes, CLASSES, iou_thresh=iou_thresh)


if __name__ == '__main__':
    main()
