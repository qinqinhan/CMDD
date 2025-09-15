import argparse
import time
import gc
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from margin import LDAMLoss
from net import FCG

from utils import AverageMeter, accuracy, log_msg, get_default_train_val_test_loader
from torch.utils.data import TensorDataset
import os
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch UEA Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='Datasets_Name')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=2, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=16, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true',
                    default=True, help='use benchmark')
parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
parser.add_argument('--noise', default='0.1', type=float,
                    help='proportion of noisy labels in the dataset')

def main():
    args = parser.parse_args()
    args.kern_size = [int(l) for l in args.kern_size.split(",")]
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_work(args)


def main_work(args):
    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date


    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    data_train, label_train,data_val,label_val, num_nodes, seq_length, num_classes,y_ori = get_default_train_val_test_loader(args)
    args.num_classes = num_classes
    val_dataset = TensorDataset(data_val, label_val)
    cls_num_list = torch.zeros(num_classes)
    for cls in range(num_classes):
        cls_num_list[cls] = (label_train == cls).sum()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # margin_loss
    criterion2 = LDAMLoss(
        cls_num_list=cls_num_list,
        device=device,
        max_m=0.5,  # 最大边际值，可调整
        s=30        # 缩放因子，可调整
    )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # training model from net.py
    model = FCG(gnn_model_type=args.arch, num_layers=args.num_layers,
                     groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size,
                     in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                     seq_len=seq_length, num_nodes=num_nodes*seq_length, num_classes=num_classes)



    # determine whether GPU or not
    if not torch.cuda.is_available():
        print("Using CPU!!!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        gc.collect()
        torch.cuda.empty_cache()

        model = model.cuda(args.gpu)
        if args.use_benchmark:
            cudnn.benchmark = True
        print('Using cudnn.benchmark.')
    else:
        print("Error!")
    criterion = nn.CrossEntropyLoss()


    # valid
    print('****************************************************')
    print(args.dataset)

    dataset_time = AverageMeter('Time', ':6.3f')
    loss_val = []
    acc_val = []




    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        current_dir,  # 当前文件所在目录
        "save_model/dataset_PenDigits_noise_0p1/best_model_PenDigits.pth"  # 模型文件名
    )
    model.load_state_dict(torch.load(model_path, map_location="cuda"))

    acc_val_per, loss_val_per, all_outputs, all_labels = validate(val_loader, model, criterion, args)
    print(acc_val_per)
    acc_val += [acc_val_per]
    loss_val += [loss_val_per]




    del model
    gc.collect()  # 强制执行垃圾回收




def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for count, (data, label) in enumerate(val_loader):

            data = data.to(device).type(torch.float)
            label = label.to(device).type(torch.long)

            output = model(data)
            loss = criterion(output, label)
            acc1 = accuracy(output, label, topk=(1, 1))


            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))


            all_outputs.append(output)
            all_labels.append(label)

    # 拼接所有批次的结果为大矩阵
    all_outputs = torch.cat(all_outputs, dim=0)  # 形状: [总样本数, 类别数]
    all_labels = torch.cat(all_labels, dim=0)  # 形状: [总样本数]

    # 返回准确率、损失以及收集的矩阵
    return top1.avg, losses.avg, all_outputs, all_labels


if __name__ == '__main__':
    main()