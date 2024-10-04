from math import cos, pi
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
from pycocotools.cocoeval import COCOeval
import math
import os
import matplotlib.pyplot as plt
from functools import partial
import torch.nn.functional as F
from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, f1_score


#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def preprocess_input(image):
    image   = image / 255
    mean    = (0.485, 0.456, 0.406)
    std     = (0.229, 0.224, 0.225)
    image   = image - mean  #去中心化
    image   = image / std  #标准化
    return image  #返回归一化后的图像数据

#   对输入图像进行resize，将图像调整为指定大小，可以选择保持纵横比（letterbox）或不保持。
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#   获得学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#   获得类
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter, lr_decay_type='cos'):
    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter

    if lr_decay_type == 'cos':
        lr = init_lr * (1 + math.cos(math.pi * current_iter / max_iter)) / 2
    elif lr_decay_type == 'step':
        # Define step_size and decay_factor according to your needs
        step_size = 10  # 每10个epoch衰减一次
        decay_factor = 0.5  # 每次衰减到原来的一半
        lr = init_lr * (decay_factor ** (epoch // step_size))
    elif lr_decay_type == 'cyclic':
        # Define base_lr, max_lr, step_size according to your needs
        base_lr = init_lr / 10
        max_lr = init_lr
        step_size = 5 * num_iter  # 半周期的迭代次数
        cycle = math.floor(1 + current_iter / (2 * step_size))
        x = abs(current_iter / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cuda'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model


def match_and_sort(pred_boxes, pred_classes, true_boxes, true_classes, iou_threshold=0.5, device='cuda'):
    # 计算IoU矩阵
    ious = calculate_iou(pred_boxes, true_boxes)  # (N, M)
    # 对每个预测框找到具有最大IoU的真实框
    max_ious, max_indices = ious.max(dim=1)
    # 根据IoU阈值筛选匹配的框
    matched_indices = [i for i, iou in enumerate(max_ious) if iou > iou_threshold]
    # 根据匹配结果对预测结果和真实标签进行排序
    sorted_pred_classes = [pred_classes[i] for i in matched_indices]
    sorted_true_classes = [true_classes[max_indices[i]] for i in matched_indices]
    sorted_pred_boxes = [pred_boxes[i] for i in matched_indices]
    sorted_true_boxes = [true_boxes[max_indices[i]] for i in matched_indices]
    # 检查列表是否为空，如果为空，则返回空张量
    if not sorted_pred_classes:
        return  (torch.empty(0, device=device), torch.empty(0, device=device),
                torch.empty(0, device=device), torch.empty(0, device=device))
    # 否则，将列表转换为张量
    sorted_pred_classes = torch.stack(sorted_pred_classes)
    sorted_true_classes = torch.stack(sorted_true_classes)
    sorted_pred_boxes = torch.stack(sorted_pred_boxes)
    sorted_true_boxes = torch.stack(sorted_true_boxes)
    return sorted_pred_classes, sorted_true_classes, sorted_pred_boxes, sorted_true_boxes


def match(pred_boxes, pred_classes, true_boxes, true_classes, iou_threshold=0.5, device='cuda'):
    pred_boxes = pred_boxes.to(device)
    true_boxes = true_boxes.to(device)
    pred_classes = pred_classes.to(device)
    true_classes = true_classes.to(device)
    # 计算IoU矩阵
    ious = calculate_iou(pred_boxes, true_boxes)  # (N, M)
    # 对每个预测框找到具有最大IoU的真实框
    max_ious, max_indices = ious.max(dim=1)
    # 根据IoU阈值筛选匹配的框
    matched_indices = [i for i, iou in enumerate(max_ious) if iou > iou_threshold]
    unmatched_indices = [i for i in range(len(true_boxes)) if i not in max_indices[matched_indices]]

    if not matched_indices:
        return torch.empty((0, pred_classes.size(1)), device=device), torch.empty(0, device=device)
    sorted_pred_classes = pred_classes[matched_indices]
    sorted_true_classes = true_classes[max_indices[matched_indices]]
    # 处理未匹配的真实框，为每个未匹配的真实框分配一个背景类（如使用一个虚构的类别-1）
    if unmatched_indices:
        # 假设背景类标签是-1，可以根据需要更改
        background_class = torch.full((len(unmatched_indices),), -1, device=device, dtype=torch.long)
        sorted_true_classes = torch.cat([sorted_true_classes, background_class], dim=0)
        # 为未匹配的真实框创建假的预测类，置信度非常低
        fake_pred_classes = torch.zeros((len(unmatched_indices), pred_classes.size(1)), device=device)
        sorted_pred_classes = torch.cat([sorted_pred_classes, fake_pred_classes], dim=0)
    #
    # # 根据匹配结果对预测结果和真实标签进行排序
    # sorted_pred_classes = [pred_classes[i] for i in matched_indices]
    # sorted_true_classes = [true_classes[max_indices[i]] for i in matched_indices]
    # # 检查列表是否为空，如果为空，则返回空张量
    # if not sorted_pred_classes:
    #     return torch.empty(0, device=device), torch.empty(0, device=device)
    # # 否则，将列表转换为张量
    # sorted_pred_classes = torch.stack(sorted_pred_classes)
    # sorted_true_classes = torch.stack(sorted_true_classes)
    return sorted_pred_classes, sorted_true_classes


# class NodeClassificationLoss(nn.Module):
#     def __init__(self, num_classes, class_weights=None, device='cuda', alpha=0.25, gamma=2.0):
#         super(NodeClassificationLoss, self).__init__()
#         self.num_classes = num_classes
#         self.device = device
#         self.class_weights = class_weights
#         self.alpha = alpha
#         self.gamma = gamma
#         if class_weights is not None:
#             self.class_weights = class_weights.to(device)
#         else:
#             self.class_weights = None
#         self.criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma, weight=self.class_weights)
#
#     def forward(self, predicted_node_classes, target_node_classes, predicted_boxes, target_boxes):
#         total_classification_loss = 0
#         for pred_node_cls, tgt_node_cls, pred_boxes, tgt_boxes in zip(predicted_node_classes, target_node_classes, predicted_boxes, target_boxes):
#             pred_boxes = pred_boxes.to(self.device)
#             tgt_boxes = tgt_boxes.to(self.device)
#             sorted_pred_node_cls, sorted_tgt_node_cls = match_and_sort(pred_boxes, pred_node_cls, tgt_boxes, tgt_node_cls)
#             if sorted_pred_node_cls.nelement() == 0 or sorted_pred_node_cls.dim() == 1:
#                 continue
#             node_classification_loss = self.criterion(sorted_pred_node_cls, sorted_tgt_node_cls.long())
#             total_classification_loss += node_classification_loss
#         num_samples = len(predicted_node_classes)
#         avg_classification_loss = total_classification_loss / max(num_samples, 1)  # 防止除以0
#         return avg_classification_loss


# 原代码 要保留
# class NodeClassificationLoss(nn.Module):
#     def __init__(self, num_classes, class_weights=None, device='cuda'):
#         super(NodeClassificationLoss, self).__init__()
#         self.num_classes = num_classes
#         self.device = device
#         self.class_weights = class_weights
#         if class_weights is not None:
#             self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
#         self.criterion = nn.NLLLoss(weight=self.class_weights, reduction='mean')  # 使用 NLLLoss
#         self.localization_loss = nn.SmoothL1Loss(reduction='mean')  # 使用 SmoothL1Loss 计算定位损失
#
#     def forward(self, predicted_node_classes, target_node_classes, predicted_boxes, target_boxes):
#         total_classification_loss = 0
#         total_localization_loss = 0
#         for pred_node_cls, tgt_node_cls, pred_boxes, tgt_boxes in zip(predicted_node_classes, target_node_classes, predicted_boxes, target_boxes):
#             pred_boxes = pred_boxes.to(self.device)
#             tgt_boxes = tgt_boxes.to(self.device)
#             # 匹配和排序框及其对应的类别
#             sorted_pred_node_cls, sorted_tgt_node_cls, sorted_pred_boxes, sorted_tgt_boxes = match_and_sort(pred_boxes, pred_node_cls, tgt_boxes, tgt_node_cls)
#             # 检查 sorted_pred_node_cls 是否为空或只有一个维度
#             if sorted_pred_node_cls.nelement() == 0 or sorted_pred_node_cls.dim() == 1:
#                 # 如果是，则跳过这个样本
#                 continue
#             # 应用log_softmax将softmax概率转换为log概率
#             log_probs = F.log_softmax(sorted_pred_node_cls, dim=1)
#             # 计算损失
#             node_classification_loss = self.criterion(log_probs, sorted_tgt_node_cls.long())
#             total_classification_loss += node_classification_loss
#             # 计算定位损失
#             localization_loss = self.localization_loss(sorted_pred_boxes, sorted_tgt_boxes)
#             total_localization_loss += localization_loss
#         # 计算平均损失
#         num_samples = len(predicted_node_classes)
#         avg_classification_loss = total_classification_loss / max(num_samples, 1)  # 防止除以0
#         avg_localization_loss = total_localization_loss / max(num_samples, 1)  # 防止除以0
#         return avg_classification_loss + avg_localization_loss

# 修改后
class NodeClassificationLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, device='cuda', localization_loss_weight=1.0):
        super(NodeClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.class_weights = class_weights
        self.localization_loss_weight = localization_loss_weight  # 权重用于平衡分类损失和定位损失
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        self.criterion = nn.NLLLoss(weight=self.class_weights, reduction='mean')  # 使用 NLLLoss
        self.localization_loss = nn.SmoothL1Loss(reduction='mean')  # 使用 SmoothL1Loss 计算定位损失

    def forward(self, predicted_node_classes, target_node_classes, predicted_boxes, target_boxes):
        total_classification_loss = 0
        total_localization_loss = 0

        # 遍历每个样本进行计算
        for pred_node_cls, tgt_node_cls, pred_boxes, tgt_boxes in zip(predicted_node_classes, target_node_classes,
                                                                      predicted_boxes, target_boxes):
            pred_boxes = pred_boxes.to(self.device)
            tgt_boxes = tgt_boxes.to(self.device)

            # 检查输入中是否有 NaN 或 Inf
            if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
                pred_boxes = torch.clamp(pred_boxes, -1e4, 1e4)
            if torch.isnan(tgt_boxes).any() or torch.isinf(tgt_boxes).any():
                tgt_boxes = torch.clamp(tgt_boxes, -1e4, 1e4)

            # 匹配和排序框及其对应的类别
            sorted_pred_node_cls, sorted_tgt_node_cls, sorted_pred_boxes, sorted_tgt_boxes = match_and_sort(pred_boxes,
                                                                                                            pred_node_cls,
                                                                                                            tgt_boxes,
                                                                                                            tgt_node_cls)

            # 检查 sorted_pred_node_cls 是否为空或只有一个维度
            if sorted_pred_node_cls.nelement() == 0 or sorted_pred_node_cls.dim() == 1:
                # 如果是，则跳过这个样本
                continue

            # 检查分类预测中是否有 NaN 或 Inf
            if torch.isnan(sorted_pred_node_cls).any() or torch.isinf(sorted_pred_node_cls).any():
                sorted_pred_node_cls = torch.clamp(sorted_pred_node_cls, -1e4, 1e4)

            # 应用log_softmax将softmax概率转换为log概率
            log_probs = F.log_softmax(sorted_pred_node_cls + 1e-8, dim=1)  # 加入1e-8偏置，防止出现极端值导致的NaN

            # 检查log_probs是否包含NaN或Inf
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                log_probs = torch.clamp(log_probs, -1e4, 1e4)

            # 计算分类损失
            node_classification_loss = self.criterion(log_probs, sorted_tgt_node_cls.long())
            total_classification_loss += node_classification_loss

            # 计算定位损失
            localization_loss = self.localization_loss(sorted_pred_boxes, sorted_tgt_boxes)
            total_localization_loss += localization_loss

        # 计算平均损失，防止除以 0
        num_samples = len(predicted_node_classes)
        avg_classification_loss = total_classification_loss / max(num_samples, 1)  # 防止除以0
        avg_localization_loss = total_localization_loss / max(num_samples, 1)  # 防止除以0

        # 返回分类损失和定位损失的和，定位损失加权
        return avg_classification_loss + self.localization_loss_weight * avg_localization_loss

# class NodeClassificationLoss(nn.Module):
#     def __init__(self, num_classes, class_weights=None, device='cuda'):
#         super(NodeClassificationLoss, self).__init__()
#         self.num_classes = num_classes
#         self.device = device
#         self.class_weights = class_weights
#         if class_weights is not None:
#             self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
#         self.criterion = nn.NLLLoss(weight=self.class_weights, reduction='mean')  # 使用 NLLLoss
#
#     def forward(self, predicted_node_classes, target_node_classes, predicted_boxes, target_boxes):
#         total_classification_loss = 0
#         for pred_node_cls, tgt_node_cls, pred_boxes, tgt_boxes in zip(predicted_node_classes, target_node_classes, predicted_boxes, target_boxes):
#             pred_boxes = pred_boxes.to(self.device)
#             tgt_boxes = tgt_boxes.to(self.device)
#             # 匹配和排序框及其对应的类别
#             sorted_pred_node_cls, sorted_tgt_node_cls = match_and_sort(pred_boxes, pred_node_cls, tgt_boxes, tgt_node_cls)
#             # 检查 sorted_pred_node_cls 是否为空或只有一个维度
#             if sorted_pred_node_cls.nelement() == 0 or sorted_pred_node_cls.dim() == 1:
#                 # 如果是，则跳过这个样本
#                 continue
#             # 应用log_softmax将softmax概率转换为log概率
#             log_probs = F.log_softmax(sorted_pred_node_cls, dim=1)
#             # 计算损失
#             node_classification_loss = self.criterion(log_probs, sorted_tgt_node_cls.long())
#             total_classification_loss += node_classification_loss
#         # 计算平均损失
#         num_samples = len(predicted_node_classes)
#         avg_classification_loss = total_classification_loss / max(num_samples, 1)  # 防止除以0
#         return avg_classification_loss

def calculate_iou(boxes1, boxes2):
    # 计算交集的左上角和右下角坐标
    intersect_tl = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    intersect_br = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    intersect_size = torch.clamp(intersect_br - intersect_tl, min=0)  # (N, M, 2)

    # 计算交集区域面积
    intersect_area = intersect_size[:, :, 0] * intersect_size[:, :, 1]  # (N, M)

    # 计算两组框的面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    # 计算并集面积
    union_area = area1[:, None] + area2 - intersect_area
    # 计算IoU
    iou = intersect_area / union_area
    return iou



# def calc_metrics(predictions, boxes, targets):
#     all_f1_scores = []
#     all_precisions = []
#     all_recalls = []
#     all_aps = []
#
#     for class_id in range(10):  # 遍历每个类别
#         tp = 0
#         fp = 0
#         fn = 0
#         all_scores = []
#         all_labels = []
#         for i, target in enumerate(targets):
#             target_boxes = target['boxes']
#             target_classes = target['classes'] - 1
#             pred_box = boxes[i]  # 40
#             pred_class = predictions[i]
#             pred_box = pred_box.to('cuda')
#             target_boxes = target_boxes.to('cuda')
#             sorted_pred_cls, sorted_tgt_cls = match(pred_box, pred_class, target_boxes, target_classes)
#             if sorted_pred_cls.nelement() == 0 or sorted_pred_cls.dim() == 1:
#                 # 如果是，则跳过这个样本
#                 continue
#             # for pred, target in zip(predictions, targets):
#             pred_probs = sorted_pred_cls
#             pred_classes = torch.argmax(pred_probs, dim=1)
#             target_classes = sorted_tgt_cls
#
#             pred_classes = pred_classes.to('cuda')
#             target_classes = target_classes.to('cuda')
#
#             true_positives = ((pred_classes == class_id) & (target_classes == class_id)).sum().item()
#             false_positives = ((pred_classes == class_id) & (target_classes != class_id)).sum().item()
#             false_negatives = ((pred_classes != class_id) & (target_classes == class_id)).sum().item()
#
#             tp += true_positives
#             fp += false_positives
#             fn += false_negatives
#
#             # 为AP计算收集所有分数和标签
#             all_scores.extend(pred_probs[:, class_id].detach().cpu().numpy())  # 将分数移至CPU并转为NumPy数组
#             all_labels.extend((target_classes == class_id).int().detach().cpu().numpy())  # 将标签移至CPU并转为NumPy数组
#             # all_scores.extend(sorted_pred_cls)
#             # all_labels.extend((target_classes == class_id).int().tolist())
#
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#
#         all_f1_scores.append(f1)
#         all_precisions.append(precision)
#         all_recalls.append(recall)
#
#         # 计算AP
#         if sum(all_labels) > 0:
#             ap = average_precision_score(all_labels, all_scores)
#         else:
#             ap = 0
#         all_aps.append(ap)
#
#     mean_f1 = sum(all_f1_scores) / len(all_f1_scores)
#     mean_ap = sum(all_aps) / len(all_aps)
#     mean_precision = sum(all_precisions) / len(all_precisions)
#     mean_recall = sum(all_recalls) / len(all_recalls)
#     # 格式化输出
#     output = "COCO results:\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {mean_ap:.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {sum([ap for ap in all_aps]) / len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {sum([ap for ap in all_aps if ap >= 0.75]) / len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {sum([ap for ap in all_aps]) / len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {sum([ap for ap in all_aps]) / len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {sum([ap for ap in all_aps]) / len(all_aps):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=   all | maxDets=100 ] = {mean_recall:.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= small | maxDets=100 ] = {sum([recall for recall in all_recalls]) / len(all_recalls):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=medium | maxDets=100 ] = {sum([recall for recall in all_recalls]) / len(all_recalls):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= large | maxDets=100 ] = {sum([recall for recall in all_recalls]) / len(all_recalls):.3f}\n"
#     output += "\nmAP(IoU=0.5) for each category:\n"
#     for i, ap in enumerate(all_aps):
#         output += f"{i + 1:02d} : {ap:.18f}\n"
#
#     print(output)
#     return mean_f1, mean_ap, mean_precision, mean_recall, all_f1_scores, all_aps, all_recalls, all_precisions

# def calc_metrics(predictions, boxes, targets):
#     all_f1_scores = []
#     all_precisions = []
#     all_recalls = []
#     all_aps = []
#     num_classes = 12
#
#     for class_id in range(num_classes):  # 遍历每个类别
#         tp = 0
#         fp = 0
#         fn = 0
#         all_scores = []
#         all_labels = []
#         for i, target in enumerate(targets):
#             target_boxes = target['boxes']
#             target_classes = target['classes'] - 1
#             pred_box = boxes[i]
#             pred_class = predictions[i]
#             pred_box = pred_box.to('cuda')
#             target_boxes = target_boxes.to('cuda')
#             sorted_pred_cls, sorted_tgt_cls = match(pred_box, pred_class, target_boxes, target_classes)
#             if sorted_pred_cls.nelement() == 0 or sorted_pred_cls.dim() == 1:
#                 continue
#             pred_probs = sorted_pred_cls
#             pred_classes = torch.argmax(pred_probs, dim=1)
#             target_classes = sorted_tgt_cls
#
#             pred_classes = pred_classes.to('cuda')
#             target_classes = target_classes.to('cuda')
#
#             true_positives = ((pred_classes == class_id) & (target_classes == class_id)).sum().item()
#             false_positives = ((pred_classes == class_id) & (target_classes != class_id)).sum().item()
#             false_negatives = ((pred_classes != class_id) & (target_classes == class_id)).sum().item()
#
#             tp += true_positives
#             fp += false_positives
#             fn += false_negatives
#
#             # 为AP计算收集所有分数和标签
#             all_scores.extend(pred_probs[:, class_id].detach().cpu().numpy())
#             all_labels.extend((target_classes == class_id).int().detach().cpu().numpy())
#
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#
#         all_f1_scores.append(f1)
#         all_precisions.append(precision)
#         all_recalls.append(recall)
#
#         # 计算AP
#         if sum(all_labels) > 0:
#             ap = average_precision_score(all_labels, all_scores)
#         else:
#             ap = 0
#         all_aps.append(ap)
#
#     mean_f1 = sum(all_f1_scores) / len(all_f1_scores)
#     mean_ap = sum(all_aps) / len(all_aps)
#     mean_precision = sum(all_precisions) / len(all_precisions)
#     mean_recall = sum(all_recalls) / len(all_recalls)
#
#     # 格式化输出
#     output = "COCO results:\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {mean_ap:.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {sum([ap for ap in all_aps])/len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {sum([ap for ap in all_aps if ap >= 0.75])/len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {sum([ap for ap in all_aps])/len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {sum([ap for ap in all_aps])/len(all_aps):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {sum([ap for ap in all_aps])/len(all_aps):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=   all | maxDets=100 ] = {mean_recall:.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= small | maxDets=100 ] = {sum([recall for recall in all_recalls])/len(all_recalls):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=medium | maxDets=100 ] = {sum([recall for recall in all_recalls])/len(all_recalls):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= large | maxDets=100 ] = {sum([recall for recall in all_recalls])/len(all_recalls):.3f}\n"
#     output += "\nmAP(IoU=0.5) for each category:\n"
#     for i, ap in enumerate(all_aps):
#         output += f"{i + 1:02d} : {ap:.18f}\n"
#
#     print(output)
#
#     return mean_f1, mean_ap, mean_precision, mean_recall, all_f1_scores, all_aps, all_recalls, all_precisions
#

def save_curves(all_labels, all_scores, class_id, output_dir):
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    ap = auc(recall, precision)

    # Save Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve for class {class_id}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f'precision_recall_curve_class_{class_id}.png'))
    plt.close()

    # Save Precision curve
    plt.figure()
    plt.plot(range(len(precision)), precision, label='Precision')
    plt.xlabel('Samples')
    plt.ylabel('Precision')
    plt.title(f'Precision curve for class {class_id}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f'precision_curve_class_{class_id}.png'))
    plt.close()

    # Save Recall curve
    plt.figure()
    plt.plot(range(len(recall)), recall, label='Recall')
    plt.xlabel('Samples')
    plt.ylabel('Recall')
    plt.title(f'Recall curve for class {class_id}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f'recall_curve_class_{class_id}.png'))
    plt.close()
def calc_metrics(predictions, boxes, targets):
    num_classes = 10
    iou_thresholds = [0.5, 0.75]
    ap_results = {iou: [] for iou in iou_thresholds}
    precision_results = {iou: [] for iou in iou_thresholds}
    recall_results = {iou: [] for iou in iou_thresholds}

    all_aps_per_class = [[] for _ in range(num_classes)]  # 存储每个类别的AP
    output_dir = 'map_13'
    os.makedirs(output_dir, exist_ok=True)
    for iou_threshold in iou_thresholds:
        all_f1_scores = []
        all_precisions = []
        all_recalls = []
        all_aps = []

        for class_id in range(num_classes):  # 遍历每个类别
            tp = 0
            fp = 0
            fn = 0
            all_scores = []
            all_labels = []
            for i, target in enumerate(targets):
                target_boxes = target['boxes']
                target_classes = target['classes'] - 1
                pred_box = boxes[i]
                pred_class = predictions[i]
                pred_box = pred_box.to('cuda')
                target_boxes = target_boxes.to('cuda')
                sorted_pred_cls, sorted_tgt_cls = match(pred_box, pred_class, target_boxes, target_classes,
                                                        iou_threshold=iou_threshold)
                if sorted_pred_cls.nelement() == 0 or sorted_tgt_cls.nelement() == 0:
                    continue
                pred_probs = sorted_pred_cls
                pred_classes = torch.argmax(pred_probs, dim=1)
                target_classes = sorted_tgt_cls

                pred_classes = pred_classes.to('cuda')
                target_classes = target_classes.to('cuda')

                true_positives = ((pred_classes == class_id) & (target_classes == class_id)).sum().item()
                false_positives = ((pred_classes == class_id) & (target_classes != class_id)).sum().item()
                false_negatives = ((pred_classes != class_id) & (target_classes == class_id)).sum().item()

                tp += true_positives
                fp += false_positives
                fn += false_negatives

                all_scores.extend(pred_probs[:, class_id].detach().cpu().numpy())
                all_labels.extend((target_classes == class_id).int().detach().cpu().numpy())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            all_f1_scores.append(f1)
            all_precisions.append(precision)
            all_recalls.append(recall)

            if sum(all_labels) > 0:
                precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
                ap = auc(recall_curve, precision_curve)
                save_curves(all_labels, all_scores, class_id, output_dir)  # 保存曲线图
            else:
                ap = 0
            all_aps.append(ap)
            all_aps_per_class[class_id].append(ap)  # 存储每个类别的AP

        mean_f1 = sum(all_f1_scores) / len(all_f1_scores)
        mean_precision = sum(all_precisions) / len(all_precisions)
        mean_recall = sum(all_recalls) / len(all_recalls)
        mean_ap = sum(all_aps) / len(all_aps)

        ap_results[iou_threshold] = all_aps  # 存储所有类别的AP列表
        precision_results[iou_threshold] = all_precisions
        recall_results[iou_threshold] = all_recalls

    mean_ap_all_thresholds = sum([sum(aps) for aps in ap_results.values()]) / (len(ap_results) * num_classes)

    output = "COCO results:\n"
    output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {mean_ap_all_thresholds:.3f}\n"
    output += f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {sum(ap_results[0.5]) / num_classes:.3f}\n"
    output += f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {sum(ap_results[0.75]) / num_classes:.3f}\n"
    output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=   all | maxDets=100 ] = {sum(recall_results[0.5]) / num_classes:.3f}\n"
    output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= small | maxDets=100 ] = {sum(recall_results[0.5]) / num_classes:.3f}\n"
    output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=medium | maxDets=100 ] = {sum(recall_results[0.5]) / num_classes:.3f}\n"
    output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= large | maxDets=100 ] = {sum(recall_results[0.5]) / num_classes:.3f}\n"
    output += "\nmAP(IoU=0.5) for each category:\n"
    for i, ap in enumerate(all_aps_per_class):
        output += f"{i + 1:02d} : {sum(ap) / len(ap):.18f}\n"  # 计算每个类别的平均AP

    print(output)

    return mean_f1, mean_ap_all_thresholds, mean_precision, mean_recall, all_f1_scores, all_aps_per_class, recall_results, precision_results





# def calc_metrics(predictions, boxes, targets):
#     num_classes = 10
#     iou_thresholds = [0.5, 0.75]
#     ap_results = {iou: [] for iou in iou_thresholds}
#     precision_results = {iou: [] for iou in iou_thresholds}
#     recall_results = {iou: [] for iou in iou_thresholds}
#
#     all_aps_per_class = [[] for _ in range(num_classes)]  # 存储每个类别的AP
#     all_true_classes = []
#     all_pred_classes = []
#
#     for iou_threshold in iou_thresholds:
#         all_f1_scores = []
#         all_precisions = []
#         all_recalls = []
#         all_aps = []
#
#         for class_id in range(num_classes):  # 遍历每个类别
#             tp = 0
#             fp = 0
#             fn = 0
#             all_scores = []
#             all_labels = []
#             for i, target in enumerate(targets):
#                 target_boxes = target['boxes']
#                 target_classes = target['classes'] - 1
#                 pred_box = boxes[i]
#                 pred_class = predictions[i]
#                 pred_box = pred_box.to('cuda')
#                 target_boxes = target_boxes.to('cuda')
#                 sorted_pred_cls, sorted_tgt_cls = match(pred_box, pred_class, target_boxes, target_classes,
#                                                         iou_threshold=iou_threshold)
#                 if sorted_pred_cls.nelement() == 0 or sorted_tgt_cls.nelement() == 0:
#                     continue
#                 pred_probs = sorted_pred_cls
#                 pred_classes = torch.argmax(pred_probs, dim=1)
#                 target_classes = sorted_tgt_cls
#
#                 pred_classes = pred_classes.to('cuda')
#                 target_classes = target_classes.to('cuda')
#
#                 true_positives = ((pred_classes == class_id) & (target_classes == class_id)).sum().item()
#                 false_positives = ((pred_classes == class_id) & (target_classes != class_id)).sum().item()
#                 false_negatives = ((pred_classes != class_id) & (target_classes == class_id)).sum().item()
#
#                 tp += true_positives
#                 fp += false_positives
#                 fn += false_negatives
#
#                 all_scores.extend(pred_probs[:, class_id].detach().cpu().numpy())
#                 all_labels.extend((target_classes == class_id).int().detach().cpu().numpy())
#
#                 all_true_classes.extend(target_classes.detach().cpu().numpy())
#                 all_pred_classes.extend(pred_classes.detach().cpu().numpy())
#
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#
#             all_f1_scores.append(f1)
#             all_precisions.append(precision)
#             all_recalls.append(recall)
#
#             if sum(all_labels) > 0:
#                 precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
#                 ap = auc(recall_curve, precision_curve)
#             else:
#                 ap = 0
#             all_aps.append(ap)
#             all_aps_per_class[class_id].append(ap)  # 存储每个类别的AP
#
#         mean_f1 = sum(all_f1_scores) / len(all_f1_scores)
#         mean_precision = sum(all_precisions) / len(all_precisions)
#         mean_recall = sum(all_recalls) / len(all_recalls)
#         mean_ap = sum(all_aps) / len(all_aps)
#
#         ap_results[iou_threshold].append(mean_ap)
#         precision_results[iou_threshold].append(mean_precision)
#         recall_results[iou_threshold].append(mean_recall)
#
#     mean_ap_all_thresholds = sum([sum(aps) for aps in ap_results.values()]) / (len(ap_results) * len(ap_results[0.5]))
#
#     # 计算多分类任务的评估指标
#     accuracy = accuracy_score(all_true_classes, all_pred_classes)
#     macro_f1 = f1_score(all_true_classes, all_pred_classes, average='macro')
#     micro_f1 = f1_score(all_true_classes, all_pred_classes, average='micro')
#     weighted_f1 = f1_score(all_true_classes, all_pred_classes, average='weighted')
#
#     output = "COCO results:\n"
#     output += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {mean_ap_all_thresholds:.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {sum(ap_results[0.5]) / len(ap_results[0.5]):.3f}\n"
#     output += f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {sum(ap_results[0.75]) / len(ap_results[0.75]):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=   all | maxDets=100 ] = {sum(recall_results[0.5]) / len(recall_results[0.5]):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= small | maxDets=100 ] = {sum(recall_results[0.5]) / len(recall_results[0.5]):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area=medium | maxDets=100 ] = {sum(recall_results[0.5]) / len(recall_results[0.5]):.3f}\n"
#     output += f"Average Recall (AR) @[ IoU=0.50:0.95   | area= large | maxDets=100 ] = {sum(recall_results[0.5]) / len(recall_results[0.5]):.3f}\n"
#     output += "\nmAP(IoU=0.5) for each category:\n"
#     for i, ap in enumerate(all_aps_per_class):
#         output += f"{i + 1:02d} : {sum(ap) / len(ap):.18f}\n"  # 计算每个类别的平均AP
#
#     output += "\nClassification results:\n"
#     output += f"Accuracy: {accuracy:.3f}\n"
#     output += f"Macro F1 Score: {macro_f1:.3f}\n"
#     output += f"Micro F1 Score: {micro_f1:.3f}\n"
#     output += f"Weighted F1 Score: {weighted_f1:.3f}\n"
#
#     print(output)
#
#     return mean_f1, mean_ap_all_thresholds, mean_precision, mean_recall, all_f1_scores, all_aps_per_class, recall_results, precision_results, accuracy, macro_f1, micro_f1, weighted_f1


def teeth_infolist(list):
    infostr = {'01: {:.2f} 02: {:.2f} 03: {:.2f} 04: {:.2f} 05: {:.2f} 06: {:.2f} 07: {:.2f} 08: {:.2f} 09: {:.2f} 10: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9])}
    return infostr
