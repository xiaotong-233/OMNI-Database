import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from model.backbone import fasterrcnn
from model.MEFL import MEFARG
from dataset import *
from util.utils import *
from conf import get_config, set_logger, set_outdir2, set_env
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_dataloader(conf):
    print('==> Preparing data...')
    train_annotation_path = 'number_train_data.txt'
    val_annotation_path = 'number_val_data.txt'
    if conf.dataset == 'tooth':
        with open(train_annotation_path) as f:
            train_lines = f.readlines()
        with open(val_annotation_path) as f:
            val_lines = f.readlines()
        trainset = ONMI(train_lines, train=True, val=False)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn, persistent_workers=True)
        valset = OMNI(val_lines, train=False, val=True)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn, persistent_workers=True)
    return train_loader, val_loader, len(trainset), len(valset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion, model_path):
    first_stage_model, model_process = fasterrcnn(model_path)
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        first_stage_model.load_state_dict(checkpoint['state_dict'])
    else:
        first_stage_model.load_state_dict(checkpoint)
    first_stage_model.eval()
    losses = AverageMeter()
    neighbor_num = 5
    net.train()
    train_loader_len = len(train_loader)
    scaler = torch.cuda.amp.GradScaler()  # 初始化梯度缩放器
    train_losses = []
    all_features = []
    all_labels = []
    for batch_idx, (inputs, label_tensors) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len, conf.lr_decay_type)
        processed_outputs, cropped_images = [], []
        for image, target in zip(inputs, label_tensors):
            processed_output, cropped_image = model_process([image], [target], training=False)
            num_boxes = processed_output[0]['boxes'].size(0)
            if num_boxes < neighbor_num:
               continue
            processed_outputs.append(processed_output)
            cropped_images.append(cropped_image)
        inputs = inputs.float()
        boxes_batch = [label_tensor[0]['boxes'].clone().detach() for label_tensor in processed_outputs]
        box_batch = [label_tensor['boxes'].clone().detach() for label_tensor in label_tensors]
        classes_batch = [label_tensor['classes'].clone().detach() - 1 for label_tensor in label_tensors]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            boxes_batch = [boxes.cuda() for boxes in boxes_batch]
            box_batch = [box.cuda() for box in box_batch]
            classes_batch = [classes.cuda() for classes in classes_batch]
        optimizer.zero_grad()
        # 使用混合精度
        with torch.cuda.amp.autocast():
            outputs = net(inputs, cropped_images)
            cl = outputs
            loss = criterion(cl, classes_batch, boxes_batch, box_batch)

        scaler.scale(loss).backward()  # 梯度缩放

        # 进行梯度裁剪，防止梯度爆炸
        scaler.unscale_(optimizer)  # 在进行梯度裁剪之前取消缩放
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 将梯度裁剪到1.0的范数内

        scaler.step(optimizer)  # 使用梯度缩放的梯度来更新模型参数
        scaler.update()  # 更新梯度缩放器

        losses.update(loss.data.item(), inputs.size(0))  # 更新损失统计
        train_losses.append(loss.item())
    return losses.avg

def val(net, val_loader, criterion, model_path, evaluate_metrics=False):
    first_stage_model, model_process = fasterrcnn(model_path)
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        first_stage_model.load_state_dict(checkpoint['state_dict'])
    else:
        first_stage_model.load_state_dict(checkpoint)
    first_stage_model.eval()
    losses = AverageMeter()
    neighbor_num = 5
    net.eval()
    all_predictions = []  # 用于存储所有预测结果
    all_boxes = []
    all_targets = []  # 用于存储所有真实标签
    val_losses = []
    all_features = []
    all_labels = []
    for batch_idx, (inputs, label_tensors) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            processed_outputs, cropped_images = [], []
            for image, target in zip(inputs, label_tensors):
                processed_output, cropped_image = model_process([image], [target], training=False)
                # 判断框的样本数是否小于neighbor_num，如果是，则跳过这个样本
                num_boxes = processed_output[0]['boxes'].size(0)
                if num_boxes < neighbor_num:
                    continue
                processed_outputs.append(processed_output)
                cropped_images.append(cropped_image)
            inputs = inputs.float()
            boxes_batch = [label_tensor[0]['boxes'].clone().detach() for label_tensor in processed_outputs]
            box_batch = [label_tensor['boxes'].clone().detach() for label_tensor in label_tensors]
            classes_batch = [label_tensor['classes'].clone().detach() - 1 for label_tensor in label_tensors]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                boxes_batch = [boxes.cuda() for boxes in boxes_batch]
                box_batch = [box.cuda() for box in box_batch]
                classes_batch = [classes.cuda() for classes in classes_batch]
            outputs = net(inputs, cropped_images)
            cl = outputs
            loss = criterion(cl, classes_batch, boxes_batch, box_batch)
            losses.update(loss.data.item(), inputs.size(0))  # 更新损失统计
            val_losses.append(loss.item())
            # 逐个目标生成预测
            batch_predictions = []
            batch_boxes = []
            for predicted_classes, predicted_boxes in zip(cl, boxes_batch):
                batch_predictions.append(predicted_classes)
                batch_boxes.append(predicted_boxes)
            filtered_indices = [i for i, output in enumerate(processed_outputs) if output[0]['boxes'].size(0) >= neighbor_num]
            all_predictions.extend([batch_predictions[i] for i in filtered_indices])
            all_boxes.extend([batch_boxes[i]for i in filtered_indices])
            all_targets.extend([label_tensors[i] for i in filtered_indices])
         
    mean_loss = losses.avg
    if evaluate_metrics:
        mean_f1, mean_ap, mean_precision, mean_recall, all_f1_scores, all_aps, all_recalls, all_precisions = calc_metrics(all_predictions, all_boxes, all_targets)
    else:
        mean_f1 = mean_ap = mean_precision = mean_recall = 0
        all_f1_scores = all_aps = all_recalls = all_precisions = []

    return mean_loss, mean_f1, mean_ap, mean_precision, mean_recall, all_f1_scores, all_aps, all_recalls, all_precisions


def main(conf):
    start_epoch = 0
    model_path = ''
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    num_classes = 10
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', 'train_weights_all10.txt')))
    logging.info("train_data_num: {}".format(train_data_num))
    net = MEFARG(num_classes=conf.num_classes, backbone1=conf.arc1, backbone2=conf.arc2)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {}".format(conf.resume))
        net = load_state_dict(net, conf.resume)
    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()
    criterion = NodeClassificationLoss(num_classes, train_weight)
    optimizer_type = conf.optimizer_type
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=conf.weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=conf.learning_rate, betas=(0.9, 0.999),
                               weight_decay=conf.weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=conf.learning_rate, betas=(0.9, 0.999),
                                weight_decay=conf.weight_decay)
    else:
        raise ValueError("Unsupported optimizer type provided!")

    print('the init learning rate is ', conf.learning_rate)

    train_losses = []
    val_losses = []
    # train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        print('Start Train')
        train_loss = train(conf, net, train_loader, optimizer, epoch, criterion, model_path)
        train_losses.append(train_loss)
        print('Finish Train' + '\n')

        # 每轮都进行验证以获取验证损失
        print('Start Validation')
        evaluate_metrics = (epoch + 1) % 5 == 0  # 每隔5轮评估一次指标
        val_loss, val_mean_f1, val_mean_ap, val_mean_recall, val_mean_precision, val_f1_scores, val_aps, val_recalls, val_precisions = val(net, val_loader, criterion, model_path, evaluate_metrics=evaluate_metrics)
        val_losses.append(val_loss)

        infostr = 'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}'.format(epoch + 1, train_loss, val_loss)
        logging.info(infostr)

        if evaluate_metrics:
            infostr += '  val_mean_f1 {:.2f}, val_mean_ap {:.2f}, val_mean_recall {:.2f}, val_mean_precision {:.2f}'.format(100. * val_mean_f1, 100. * val_mean_ap, 100. * val_mean_recall, 100. * val_mean_precision)
            logging.info(infostr)

        if (epoch + 1) % 5 == 0:
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


