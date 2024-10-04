import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from model.ANFL import MEFARG
from dataset import *
from util.utils import *
from conf import get_config, set_logger, set_outdir2, set_env
from torch.nn.utils.rnn import pad_sequence

def get_dataloader(conf):
    print('==> Preparing data...')
    train_annotation_path = '/mnt/disk1/data0/jxt/dataset/data/resized/list/number_train_data.txt'
    val_annotation_path = '/mnt/disk1/data0/jxt/dataset/data/resized/list/number_val_data.txt'
    if conf.dataset == 'tooth':
        with open(train_annotation_path) as f:
            train_lines = f.readlines()
        with open(val_annotation_path) as f:
            val_lines = f.readlines()
        trainset = BP4D(train_lines, train=True, val=False)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn)
        valset = BP4D(val_lines, train=False, val=True)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn)

    return train_loader, val_loader, len(trainset), len(valset)

def adjust_iou_threshold(current_epoch, total_epochs):
    # 根据当前轮次和总轮次来动态调整 iou_threshold
    # 以下是一个简单的线性调整示例，你可以根据需要进行修改
    initial_iou_threshold = 0.01
    final_iou_threshold = 0.5
    delta = (final_iou_threshold - initial_iou_threshold) / total_epochs
    return initial_iou_threshold + delta * current_epoch
# Train
# 每个标注为一行
def train(conf, net, train_loader, optimizer, epoch, criterion):
    # 动态调整 iou_threshold
    iou_threshold = adjust_iou_threshold(epoch, conf.epochs)
    losses = AverageMeter()
    local_losses = AverageMeter()
    class_losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs, label_tensors) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        inputs = inputs.float()
        boxes_batch = [label_tensor['boxes'] for label_tensor in label_tensors]
        classes_batch = [label_tensor['labels'] - 1 for label_tensor in label_tensors]
        # # 使用 pad_sequence 填充成相同形状
        # boxes_tensor = pad_sequence(boxes_batch, batch_first=True, padding_value=-1)
        # classes_tensor = pad_sequence(classes_batch, batch_first=True, padding_value=-1)
        # boxes_tensor = torch.nn.functional.pad(boxes_tensor, (0, 0, 0, max(0, 32 - boxes_tensor.size(1))), value=-1)
        # classes_tensor = torch.nn.functional.pad(classes_tensor, (0, max(0, 32 - classes_tensor.size(1))), value=-1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            boxes_batch = [boxes.cuda() for boxes in boxes_batch]
            classes_batch = [classes.cuda() for classes in classes_batch]
        optimizer.zero_grad()
        # for label in label_tensors:
        #     boxes_tensor = torch.tensor(label['boxes'])
        #     classes_tensor = torch.tensor(label['classes'])
        outputs = net(inputs, boxes_batch)
        cl, anchor = outputs
        loss, local_loss, class_loss = criterion(anchor, cl, boxes_batch, classes_batch, iou_threshold)
        loss.backward()  # 在整个批次上进行反向传播
        optimizer.step()  # 在整个批次上进行优化
        losses.update(loss.data.item(), inputs.size(0) * len(label_tensors))  # 更新损失统计
        local_losses.update(local_loss.data.item(), inputs.size(0) * len(label_tensors))
        class_losses.update(class_loss.data.item(), inputs.size(0) * len(label_tensors))
    return losses.avg, local_losses.avg, class_losses.avg
def val(net, val_loader, optimizer, epoch, criterion):
    # 动态调整 iou_threshold
    criterion.iou_threshold = adjust_iou_threshold(epoch, conf.epochs)
    losses = AverageMeter()
    local_losses = AverageMeter()
    class_losses = AverageMeter()
    net.eval()
    val_loader_len = len(val_loader)
    all_predictions = []  # 用于存储所有预测结果
    all_targets = []  # 用于存储所有真实标签
    for batch_idx, (inputs, label_tensors) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, val_loader_len)
            inputs = inputs.float()
            boxes_batch = [label_tensor['boxes'] for label_tensor in label_tensors]
            classes_batch = [label_tensor['labels'] - 1 for label_tensor in label_tensors]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                boxes_batch = [boxes.cuda() for boxes in boxes_batch]
                classes_batch = [classes.cuda() for classes in classes_batch]
            optimizer.zero_grad()
            outputs = net(inputs, boxes_batch)
            cl, anchor = outputs
            loss, local_loss, class_loss = criterion(anchor, cl, boxes_batch, classes_batch)
            # 逐个目标生成预测
            batch_predictions = []
            for predicted_boxes, predicted_classes in zip(anchor, cl):
                # predicted_boxes = anchor[i] # 假设anchor即为预测的边界框
                # predicted_classes = cl[i]  # 假设cl为类别预测概率，取最大概率的类别作为预测结果
                batch_predictions.append({'boxes': predicted_boxes, 'labels': predicted_classes})
            all_predictions.extend(batch_predictions)
            all_targets.extend(label_tensors)
            losses.update(loss.data.item(), inputs.size(0))  # 更新损失统计
            local_losses.update(local_loss.data.item(), inputs.size(0))
            class_losses.update(class_loss.data.item(), inputs.size(0))
    mean_loss = losses.avg
    mean_local_losses = local_losses.avg
    mean_class_losses = class_losses.avg
    mean_f1, mean_ap, mean_precision, mean_recall = calc_metrics(all_predictions, all_targets)
    return mean_loss, mean_local_losses, mean_class_losses, mean_f1, mean_ap, mean_precision, mean_recall

def main(conf):
    start_epoch = 0
    # data
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    num_classes = 18
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', 'train_weights.txt')))
    logging.info("train_data_num: {}".format(train_data_num))
    net = MEFARG(num_classes=conf.num_classes, neighbor_num=conf.neighbor_num, metric=conf.metric)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {}".format(conf.resume))
        net = load_state_dict(net, conf.resume)
    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()
    criterion = DetectionAndNodeClassificationLoss(num_classes, train_weight)
    # criterion = DetectionLoss(weight=train_weight)
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)
    # train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        print('Start Train')
        train_loss, train_local_loss, train_class_loss = train(conf, net, train_loader, optimizer, epoch, criterion)
        print('Finish Train'+'\n')
        print('Start Validation')
        val_loss, val_local_loss, val_class_loss, val_mean_f1, val_mean_ap, val_mean_recall, val_mean_precision = val(net, val_loader, optimizer, epoch, criterion)
        infostr = 'Epoch:  {}   train_loss: {:.5f}  train_local_loss: {:.5f}  train_class_loss: {:.5f}  val_loss: {:.5f}  val_local_loss: {:.5f}  val_class_loss: {:.5f}  val_mean_f1 {:.2f},val_mean_ap {:.2f},val_mean_recall {:.2f},val_mean_precision {:.2f}'.format(epoch + 1, train_loss, train_local_loss, train_class_loss, val_loss, val_local_loss, val_class_loss, 100. * val_mean_f1, 100. * val_mean_ap, 100. * val_mean_recall, 100. * val_mean_precision)
        logging.info(infostr)
        # infostr = 'F1-score-list:\n{}'.format(dataset_info(val_f1))
        # logging.info(infostr)
        # infostr = 'Ap-list:\n{}'.format(dataset_info(val_ap))
        # logging.info(infostr)
        # infostr = 'Recall-list:\n{}'.format(dataset_info(val_recall))
        # logging.info(infostr)
        # infostr = 'Precision-list:\n{}'.format(dataset_info(val_precision))
        # logging.info(infostr)
        # save checkpoints
        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model.pth'))

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model.pth'))


# ---------------------------------------------------------------------------------

if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir2(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

