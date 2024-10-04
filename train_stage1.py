import os
import numpy as np
import torch
import torch.nn as nn
from model.backbone import fasterrcnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from model.ANFL import MEFARG
from dataset import *
from util.utils import *
from conf import get_config, set_logger, set_outdir1, set_env
from loss_fasterrcnn import *

def get_dataloader(conf):
    print('==> Preparing data...')
    train_annotation_path = 'number_train_data.txt'
    val_annotation_path = 'number_val_data.txt'
    if conf.dataset == 'tooth':
        with open(train_annotation_path) as f:
            train_lines = f.readlines()
        with open(val_annotation_path) as f:
            val_lines = f.readlines()
        trainset = OMNI(train_lines, train=True, val=False)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn)
        valset = OMNI(val_lines, train=False, val=True)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn)

    return train_loader, val_loader, len(trainset), len(valset)

def modify_labels(targets):
    modified_targets = []
    for target in targets:
        boxes = target['boxes']
        labels = torch.ones(boxes.shape[0], dtype=torch.int64)  # 将所有标签设置为1（牙齿）
        modified_targets.append({'boxes': boxes, 'labels': labels})
    return modified_targets
    
# Train
# 每个标注为一行
def train(conf, net, model_process, train_loader, optimizer, epoch):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs, label_tensors) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        inputs = inputs.float()
        modified_label_tensors = modify_labels(label_tensors)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            for label_tensor in modified_label_tensors:
                label_tensor['boxes'] = label_tensor['boxes'].cuda()
                label_tensor['labels'] = label_tensor['labels'].cuda()
        optimizer.zero_grad()
        loss_dicts = model_process(inputs, modified_label_tensors, training=True)  # 使用 model_process 处理数据
        total_loss = torch.sum(torch.stack([sum(loss_dict.values()) for loss_dict in loss_dicts]))
        total_loss.backward()  # Perform backpropagation on the average loss
        optimizer.step()
        losses.update(total_loss.item(), inputs.size(0))  # Update the losses
    mean_loss = losses.avg
    with open('results/loss_record.txt', 'a') as f:
        f.write(f"Epoch {epoch}: {mean_loss}\n")
    return mean_loss
    
def val(net, model_process, val_loader, epoch):
    net.eval()
    losses = AverageMeter()
    total_loss = 0
    all_predictions = []  # 用于存储所有预测结果
    all_targets = []  # 用于存储所有真实标签
    for batch_idx, (inputs, label_tensors) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs = inputs.float()
            modified_label_tensors = modify_labels(label_tensors)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                for label_tensor in modified_label_tensors:
                    label_tensor['boxes'] = label_tensor['boxes'].cuda()
                    label_tensor['labels'] = label_tensor['labels'].cuda()
            results, local_features_list = model_process(inputs, modified_label_tensors, training=False)  # 获取检测结果
            batch_loss = calculate_loss(results, modified_label_tensors)
            total_loss += batch_loss
            losses.update(total_loss.item(), inputs.size(0))
            all_predictions.extend(results)
            all_targets.extend(modified_label_tensors)
    metrics = calc_metrics(all_predictions, all_targets)
    mean_f1 = metrics['F1 Score']
    mean_iou = metrics['mIoU']
    mean_precision = metrics['Precision']
    mean_recall = metrics['Recall']
    with open('results/loss_record.txt', 'a') as f:
        f.write(f"Epoch {epoch}: {losses.avg}\n")
    return losses.avg, mean_f1, mean_iou, mean_precision, mean_recall

def main(conf):
    start_epoch = 0
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    logging.info("train_data_num: {}".format(train_data_num))
    net, model_process = fasterrcnn(model_path=None)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {}".format(conf.resume))
        net = load_state_dict(net, conf.resume)
    optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)
    # Initialize best_val_loss to a large value
    best_val_loss = float('inf')
    # train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        print('Start Train')
        train_loss = train(conf, net, model_process, train_loader, optimizer, epoch)
        print('Finish Train'+'\n')
        print('Start Validation')
        val_loss, val_mean_f1, val_mean_iou, val_mean_recall, val_mean_precision = val(net, model_process, val_loader, epoch)
        infostr = 'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1 {:.2f},val_mean_iou {:.2f},val_mean_recall {:.2f},val_mean_precision {:.2f}'.format(epoch + 1, train_loss, val_loss, 100. * val_mean_f1, 100. * val_mean_iou, 100. * val_mean_recall, 100. * val_mean_precision)
        logging.info(infostr)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(best_checkpoint, os.path.join(conf['outdir'], 'best_model.pth'))
            logging.info("Best model saved with val_loss: {:.5f}".format(best_val_loss))

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


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir1(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
