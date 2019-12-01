import argparse
import os
import os.path as osp
import numpy as np
#from tqdm import tqdm

import torch

from mypath import Path
#from dataloaders import make_data_loader
from dataloaders.datasets import leadbang
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.utils import data

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
   
   
def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(0.01, i_iter, 1000, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    return lr

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        self.train_set = leadbang.LeadBangTrain("/leadbang/data/")
        self.train_loader = data.DataLoader(self.train_set, 
                    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        # self.val_loader = leadbang.LeadBangTest(Path.db_root_dir("leadbang"))
        self.nclass = 2

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        
        weight = torch.FloatTensor([1, 2])
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        

        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()

        args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        # tbar = tqdm(self.train_loader)
        # num_img_tr = len(self.train_loader)
        for i, sample in enumerate(self.train_loader):
            lr = adjust_learning_rate(self.optimizer, epoch)
            if i == 0:
                print("learning rate: ", lr)
            image, target, _, _ = sample
            
            image, target = image.cuda(), target.cuda()
            
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            avg_loss = train_loss / (i + 1)
            if i % 10 == 9:
                print('>epoch-{} iter-{}, loss {}'.format(epoch, i, avg_loss))

        if epoch % 10 == 0:
            print('taking snapshot ...')
            torch.save(self.model.state_dict(),osp.join('checkpoint', str(epoch)+'.pth'))     


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    
    args.cuda = True
    args.gpu_ids = [0,2]

    # if args.sync_bn is None:
    #     if args.cuda and len(args.gpu_ids) > 1:
    #         args.sync_bn = True
    #     else:
    #         args.sync_bn = False
    args.sync_bn = True

    # default settings for epochs, batch_size and lr
    args.epochs = 1000

    # if args.batch_size is None:
    #     args.batch_size = 4 * len(args.gpu_ids)

    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size

    # if args.lr is None:
    #     lrs = {
    #         'coco': 0.1,
    #         'cityscapes': 0.01,
    #         'pascal': 0.007,
    #     }
    #     args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size
    args.lr = 0.01
    args.batch_size = 4


    print(args)
    torch.manual_seed(args.seed)
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    trainer = Trainer(args)
    for epoch in range(args.epochs):
        trainer.training(epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        #     trainer.validation(epoch)


if __name__ == "__main__":
   main()
