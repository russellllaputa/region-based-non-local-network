# Code for paper:
# [Title]  - "Region-based Non-local Operation for Video Classification"
# [Author] - Guoxi Huang, Adrian G. Bors
# [Github] - https://github.com/guoxih/region-based-non-local-network

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool
from torchsummary import summary
from tensorboardX import SummaryWriter



import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


parser = argparse.ArgumentParser(description='PyTorch Kinetics Training')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32, the total bach size of a node)')
parser.add_argument('--batch_multiplier', default=1, type=int, metavar='N')
parser.add_argument('--use_warmup', default=False, action="store_true")
parser.add_argument('--warmup_epochs', default=1, type=int, metavar='N')
parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--dataset', type=str)
parser.add_argument('--modality', type=str, default='RGB', help='RGB or Flow')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--lr_type', default='step', type=str, metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--root-model', default='checkpoint', type=str)
parser.add_argument('--clip-gradient', '--gd', default=None, type=float, metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 5)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--suffix', type=str, default='1')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://144.32.50.212:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--available_gpus', default='0, 1, 2, 3', type=str)
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--eval-freq', '-ef', default=1, type=int, metavar='N', help='evaluation frequency (default: 5)')

best_acc1 = 0
global global_time
global_time = time.time()


def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.available_gpus
    args.consensus_type = 'avg'
    args.pretrain = 'imagenet'
    args.tune_from = None
    args.img_feature_dim = 256
    args.loss_type ='nll'
    args.evaluate = False

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

#--------------------------------------------------------------------------------------------------------------------------

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=rank)
    else:
        rank = 0
    # create model
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    args.store_name += '_lr{}'.format(args.lr)
    args.store_name += '_wd{:.1e}'.format(args.weight_decay)
    args.store_name += '_do{}'.format(args.dropout)
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders(args, rank)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)
    
    # first synchronization of initial weights
    # sync_initial_weights(model, args.rank, args.world_size)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if rank == 0:
        print(model)
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have on a node
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.start_epoch == 1:
                args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
#             if args.gpu is not None:
#                 # best_acc1 may be from a checkpoint from a different GPU
#                 best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_dataset = TSNDataSet(args.dataset, args.root_path, args.train_list, num_segments=args.num_segments,
                       new_length=data_length,
                       modality=args.modality,
                       image_tmpl=prefix,
                       transform=torchvision.transforms.Compose([train_augmentation,
                           Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize]), 
                      dense_sample=args.dense_sample)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.dataset, args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    log_training = open(os.path.join(args.root_model, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_model, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_model, args.store_name))
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
#         adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer, args, rank)
        if rank % ngpus_per_node == 0:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, False, args, rank)
        
        if epoch % 5 == 0 and rank % ngpus_per_node == 0:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, False, args, rank, e=epoch)

        # evaluate on validation set
        is_best = False
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            acc1 = validate(val_loader, model, criterion, epoch, args, rank, log_training, tf_writer)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, rank)

#--------------------------------------------------------------------------------------------------------------------------

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer, args, rank):
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    model.zero_grad()
    loss_tmp = []
    acc_tmp = []
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, args, (epoch-1) + float(i) / len(train_loader))
        i += 1
#         if (i+1) % args.batch_multiplier == 0:
#             optimizer.step()
#             optimizer.zero_grad()
        # measure data loading time
        
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / args.batch_multiplier # divide batch_multiplier as grad accumulation

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item()*args.batch_multiplier, input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        loss_tmp.append(loss.item()*args.batch_multiplier)
        acc_tmp.append(prec1.item())

        # compute gradient and do SGD step
        if i % args.batch_multiplier != 0:
            if args.multiprocessing_distributed:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        else:
            loss.backward()
            if args.clip_gradient is not None:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
#             optimizer.zero_grad()
            
        if rank == 0:
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % (args.print_freq*args.batch_multiplier*5) == 0:
                tf_writer.add_scalar('loss/step', np.mean(loss_tmp), ((epoch-1)*len(train_loader)+i)/args.batch_multiplier)
                tf_writer.add_scalar('acc/step', np.mean(acc_tmp), ((epoch-1)*len(train_loader)+i)/args.batch_multiplier)
                loss_tmp = []
                acc_tmp = []
            if i % (args.print_freq * args.batch_multiplier) == 0:
                output = ('Epoch: [{0:3d}][{1:4d}/{2:4d}], lr: {lr:.5f}\t'
                          'Time {time:.1f}\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, int(i/args.batch_multiplier), int(len(train_loader)/args.batch_multiplier), time=(time.time()-global_time)/60.,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
                print(output)
                log.write(output + '\n')
                log.flush()
                if i % (args.print_freq*args.batch_multiplier*10) == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        tf_writer.add_histogram('weights/'+tag, value.detach(), (epoch*len(train_loader)+i)/args.batch_multiplier)
                        tf_writer.add_histogram('grads/'+tag, value.grad.detach().abs().mean(), (epoch*len(train_loader)+i)/args.batch_multiplier)
                        
        if i % args.batch_multiplier == 0:
            optimizer.zero_grad()
                        
    if rank == 0:
        tf_writer.add_scalar('loss/train', losses.avg, epoch-1)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch-1)
    

def validate(val_loader, model, criterion, epoch, args, rank, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            i += 1
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()
    
    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()
        
    if rank == 0:
        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses.avg, epoch-1)
            tf_writer.add_scalar('acc/test_top1', top1.avg, epoch-1)

    return top1.avg


def save_checkpoint(state, is_best, args, rank, e=None):
    if e is not None:
        filename = '%s/%s/rank%d_epoch%d_ckpt.pth.tar' % (args.root_model, args.store_name, rank, e)
    else:
        filename = '%s/%s/rank%d_ckpt.pth.tar' % (args.root_model, args.store_name, rank)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
       
        
def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps, args, epoch_float=-1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch_float < 0:
        epoch_float = float(epoch)
    if args.use_warmup and epoch_float < args.warmup_epochs:
        base_lr = args.lr / 10
        lr = (epoch_float / args.warmup_epochs) * (args.lr - base_lr) + base_lr
        decay = args.weight_decay
    elif lr_type == 'step':
        decay = 0.1 ** (sum(epoch > np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        if args.use_warmup:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch_float - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch_float / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders(args, rank):
    """Create log and model folder"""
    if rank == 0:
        folders_util = [args.root_model, os.path.join(args.root_model, args.store_name)]
        for folder in folders_util:
            if not os.path.exists(folder):
                print('creating folder ' + folder)
                os.mkdir(folder)

def sync_initial_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)



if __name__ == '__main__':
    main()
