import math
from functools import partial

import torch
import torch.nn as nn


def calc_iou(a, b):
    max_length = torch.max(a)
    a = a / max_length
    b = b / max_length

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def get_target(anchor, bbox_annotation, classification, cuda):
    # ------------------------------------------------------#
    #   计算真实框和先验框的交并比
    #   anchor              num_anchors, 4
    #   bbox_annotation     num_true_boxes, 5
    #   Iou                 num_anchors, num_true_boxes
    # ------------------------------------------------------#
    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

    # ------------------------------------------------------#
    #   计算与先验框重合度最大的真实框
    #   IoU_max             num_anchors,
    #   IoU_argmax          num_anchors,
    # ------------------------------------------------------#
    IoU_max, IoU_argmax = torch.max(IoU, dim=1)

    # ------------------------------------------------------#
    #   寻找哪些先验框在计算loss的时候需要忽略
    # ------------------------------------------------------#
    targets = torch.ones_like(classification) * -1
    targets = targets.type_as(classification)

    # ------------------------------------------#
    #   重合度小于0.4需要参与训练
    # ------------------------------------------#
    targets[torch.lt(IoU_max, 0.4), :] = 0

    # --------------------------------------------------#
    #   重合度大于0.5需要参与训练，还需要计算回归loss
    # --------------------------------------------------#
    positive_indices = torch.ge(IoU_max, 0.5)

    # --------------------------------------------------#
    #   取出每个先验框最对应的真实框
    # --------------------------------------------------#
    assigned_annotations = bbox_annotation[IoU_argmax, :]

    # --------------------------------------------------#
    #   将对应的种类置为1
    # --------------------------------------------------#
    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
    # --------------------------------------------------#
    #   计算正样本数量
    # --------------------------------------------------#
    num_positive_anchors = positive_indices.sum()
    return targets, num_positive_anchors, positive_indices, assigned_annotations


def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
    # --------------------------------------------------#
    #   取出作为正样本的先验框对应的真实框
    # --------------------------------------------------#
    assigned_annotations = assigned_annotations[positive_indices, :]

    # --------------------------------------------------#
    #   取出作为正样本的先验框
    # --------------------------------------------------#
    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

    # --------------------------------------------------#
    #   计算真实框的宽高与中心
    # --------------------------------------------------#
    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    # ---------------------------------------------------#
    #   利用真实框和先验框进行编码，获得应该有的预测结果
    # ---------------------------------------------------#
    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = torch.log(gt_widths / anchor_widths_pi)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)

    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
    targets = targets.t()
    return targets


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, alpha=0.25, gamma=2.0, cuda=True):
        # ---------------------------#
        #   获得batch_size的大小
        # ---------------------------#
        batch_size = classifications.shape[0]

        # --------------------------------------------#
        #   获得先验框，将先验框转换成中心宽高的形式
        # --------------------------------------------#
        dtype = regressions.dtype
        anchor = anchors[0, :, :].to(dtype)
        # --------------------------------------------#
        #   将先验框转换成中心，宽高的形式
        # --------------------------------------------#
        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        regression_losses = []
        classification_losses = []
        for j in range(batch_size):
            # -------------------------------------------------------#
            #   取出每张图片对应的真实框、种类预测结果和回归预测结果
            # -------------------------------------------------------#
            bbox_annotation = annotations[j]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            classification = torch.clamp(classification, 5e-4, 1.0 - 5e-4)

            if len(bbox_annotation) == 0:
                # -------------------------------------------------------#
                #   当图片中不存在真实框的时候，所有特征点均为负样本
                # -------------------------------------------------------#
                alpha_factor = torch.ones_like(classification) * alpha
                alpha_factor = alpha_factor.type_as(classification)

                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                # -------------------------------------------------------#
                #   计算特征点对应的交叉熵
                # -------------------------------------------------------#
                bce = - (torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                classification_losses.append(cls_loss.sum())
                # -------------------------------------------------------#
                #   回归损失此时为0
                # -------------------------------------------------------#
                regression_losses.append(torch.tensor(0).type_as(classification))

                continue

            # ------------------------------------------------------#
            #   计算真实框和先验框的交并比
            #   targets                 num_anchors, num_classes
            #   num_positive_anchors    正样本的数量
            #   positive_indices        num_anchors,
            #   assigned_annotations    num_anchors, 5
            # ------------------------------------------------------#
            targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor,
                                                                                               bbox_annotation,
                                                                                               classification, cuda)

            # ------------------------------------------------------#
            #   首先计算交叉熵loss
            # ------------------------------------------------------#
            alpha_factor = torch.ones_like(targets) * alpha
            alpha_factor = alpha_factor.type_as(classification)
            # ------------------------------------------------------#
            #   这里使用的是Focal loss的思想，
            #   易分类样本权值小
            #   难分类样本权值大
            # ------------------------------------------------------#
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = - (targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            # ------------------------------------------------------#
            #   把忽略的先验框的loss置为0
            # ------------------------------------------------------#
            zeros = torch.zeros_like(cls_loss)
            zeros = zeros.type_as(cls_loss)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            # ------------------------------------------------------#
            #   如果存在先验框为正样本的话
            # ------------------------------------------------------#
            if positive_indices.sum() > 0:
                targets = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights,
                                      anchor_ctr_x, anchor_ctr_y)
                # ---------------------------------------------------#
                #   将网络应该有的预测结果和实际的预测结果进行比较
                #   计算smooth l1 loss
                # ---------------------------------------------------#
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).type_as(classification))

        # 计算平均loss并返回
        c_loss = torch.stack(classification_losses).mean()
        r_loss = torch.stack(regression_losses).mean()
        loss = c_loss + r_loss
        return loss, c_loss, r_loss


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# import math
# from functools import partial
#
# import torch
# import torch.nn as nn
#
#
# def calc_iou(a, b):
#     max_length = torch.max(a)
#     a = a / max_length
#     b = b / max_length
#
#     area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
#     iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
#     ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
#     iw = torch.clamp(iw, min=0)
#     ih = torch.clamp(ih, min=0)
#     # ua表示并集的面积
#     ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
#     ua = torch.clamp(ua, min=1e-8)
#     # 交集的面积
#     intersection = iw * ih
#     IoU = intersection / ua
#
#     return IoU
#
# def get_target(anchor, bbox_annotation, classification, cuda):
#     #------------------------------------------------------#
#     #   计算真实框和先验框的交并比
#     #   anchor              num_anchors, 4
#     #   bbox_annotation     num_true_boxes, 5
#     #   Iou                 num_anchors, num_true_boxes
#     #------------------------------------------------------#
#     IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
#
#     #------------------------------------------------------#
#     #   计算与先验框重合度最大的真实框
#     #   IoU_max             num_anchors,
#     #   IoU_argmax          num_anchors,
#     #------------------------------------------------------#
#     IoU_max, IoU_argmax = torch.max(IoU, dim=1)
#
#     #------------------------------------------------------#
#     #   寻找哪些先验框在计算loss的时候需要忽略
#     #------------------------------------------------------#
#     targets = torch.ones_like(classification) * -1
#     targets = targets.type_as(classification)
#
#     #------------------------------------------#
#     #   重合度小于0.4需要参与训练
#     #------------------------------------------#
#     targets[torch.lt(IoU_max, 0.4), :] = 0
#
#     #--------------------------------------------------#
#     #   重合度大于0.5需要参与训练，还需要计算回归loss
#     #--------------------------------------------------#
#     positive_indices = torch.ge(IoU_max, 0.5)
#
#     #--------------------------------------------------#
#     #   取出每个先验框最对应的真实框
#     #--------------------------------------------------#
#     assigned_annotations = bbox_annotation[IoU_argmax, :]
#
#     #--------------------------------------------------#
#     #   将对应的种类置为1
#     #--------------------------------------------------#
#     targets[positive_indices, :] = 0
#     targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
#     # print(positive_indices)
#     # print(assigned_annotations)
#     # print(targets)
#     #--------------------------------------------------#
#     #   计算正样本数量
#     #--------------------------------------------------#
#     num_positive_anchors = positive_indices.sum()
#     return targets, num_positive_anchors, positive_indices, assigned_annotations
#
# # def get_target(anchor, bbox_annotation, classification, cuda):
# #     #------------------------------------------------------#
# #     #   计算真实框和先验框的交并比
# #     #   anchor              num_anchors, 4
# #     #   bbox_annotation     num_true_boxes, 5
# #     #   Iou                 num_anchors, num_true_boxes
# #     #------------------------------------------------------#
# #     IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
# #
# #     #------------------------------------------------------#
# #     #   计算与先验框重合度最大的真实框
# #     #   IoU_max             num_anchors,
# #     #   IoU_argmax          num_anchors,
# #     #------------------------------------------------------#
# #     IoU_max, IoU_argmax = torch.max(IoU, dim=1)
# #     # Ensure the correct number of anchors for the classification targets
# #     num_anchors = classification.size(0)
# #     if IoU_max.size(0) != num_anchors:
# #         IoU_max = IoU_max[:num_anchors]
# #         IoU_argmax = IoU_argmax[:num_anchors]
# #
# #         # 检查 bbox_annotation 的大小是否小于 IoU_max
# #     if bbox_annotation.size(0) < IoU_max.size(0):
# #         padding_size = IoU_max.size(0) - bbox_annotation.size(0)
# #         # 填充 0，保持 bbox_annotation 的形状一致
# #         padding = torch.zeros(padding_size, bbox_annotation.size(1)).to(bbox_annotation.device)
# #         bbox_annotation = torch.cat([bbox_annotation, padding], dim=0)
# #     #------------------------------------------------------#
# #     #   寻找哪些先验框在计算loss的时候需要忽略
# #     #------------------------------------------------------#
# #     targets = torch.ones_like(classification) * -1
# #     targets = targets.type_as(classification)
# #
# #     #------------------------------------------#
# #     #   重合度小于0.4需要参与训练
# #     #------------------------------------------#
# #     targets[torch.lt(IoU_max, 0.4), :] = 0
# #
# #     #--------------------------------------------------#
# #     #   重合度大于0.5需要参与训练，还需要计算回归loss
# #     #--------------------------------------------------#
# #     positive_indices = torch.ge(IoU_max, 0.5)
# #
# #     #--------------------------------------------------#
# #     #   取出每个先验框最对应的真实框
# #     #--------------------------------------------------#
# #     assigned_annotations = bbox_annotation[IoU_argmax, :]
# #
# #     #--------------------------------------------------#
# #     #   将对应的种类置为1
# #     #--------------------------------------------------#
# #     targets[positive_indices, :] = 0
# #     targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
# #     # print(positive_indices)
# #     # print(assigned_annotations)
# #     # print(targets)
# #     #--------------------------------------------------#
# #     #   计算正样本数量
# #     #--------------------------------------------------#
# #     num_positive_anchors = positive_indices.sum()
# #     return targets, num_positive_anchors, positive_indices, assigned_annotations
#
#
# def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
#     # 此处的 positive_indices 应该已经经过 get_target 中的维度调整
#     # 如果 assigned_annotations 的长度为 0，直接返回空结果以避免后续错误
#     if assigned_annotations.size(0) == 0:
#         return torch.zeros(0, 4).to(anchor_widths.device)
#     # Adjust positive indices if they exceed available anchors
#     positive_indices = positive_indices.nonzero(as_tuple=False).squeeze(1)
#
#     if positive_indices.shape[0] > anchor_widths.shape[0]:
#         positive_indices = positive_indices[:anchor_widths.shape[0]]
#     # 如果 positive_indices 的大小小于 anchor_widths，则填充
#     elif positive_indices.shape[0] < anchor_widths.shape[0]:
#         positive_indices = torch.cat([positive_indices, torch.zeros(anchor_widths.shape[0] - positive_indices.shape[0],
#                                                                     dtype=torch.bool).to(positive_indices.device)])
#     # 取出作为正样本的先验框
#     anchor_widths_pi = anchor_widths[positive_indices]
#     anchor_heights_pi = anchor_heights[positive_indices]
#     anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
#     anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
#
#     if assigned_annotations.size(0) < anchor_widths_pi.size(0):
#         padding_size = anchor_widths_pi.size(0) - assigned_annotations.size(0)
#         # 用零填充
#         padding = torch.zeros(padding_size, assigned_annotations.size(1)).to(assigned_annotations.device)
#         assigned_annotations = torch.cat([assigned_annotations, padding], dim=0)
#
#     # 计算真实框的宽高与中心
#     gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
#     gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
#     gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
#     gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
#
#     gt_widths = torch.clamp(gt_widths, min=1)
#     gt_heights = torch.clamp(gt_heights, min=1)
#
#     # 利用真实框和先验框进行编码，获得应该有的预测结果
#     targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
#     targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
#     targets_dw = torch.log(gt_widths / anchor_widths_pi)
#     targets_dh = torch.log(gt_heights / anchor_heights_pi)
#
#     targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
#     targets = targets.t()
#     return targets
#
# # 可能需要调
# class FocalLoss(nn.Module):
#     def __init__(self, class_weights = None):
#     # def __init__(self):
#         super(FocalLoss, self).__init__()
#         self.class_weights = class_weights   #将class_weights存储为类的属性
# # alpha为平衡参数，gamma为调节参数
# # 修改
#     def forward(self, classifications, regressions, anchors, annotations, alpha = 0.25, gamma = 2.0, cuda = True):
#     # def forward(self, classifications, regressions, anchors, annotations, alpha=0.25, gamma=2.0, cuda=True):
#         #---------------------------#
#         #   获得batch_size的大小
#         #---------------------------#
#         batch_size = classifications.shape[0]
#
#         #--------------------------------------------#
#         #   获得先验框，将先验框转换成中心宽高的形式
#         #--------------------------------------------#
#         dtype = regressions.dtype
#         anchor = anchors[0, :, :].to(dtype)
#         #--------------------------------------------#
#         #   将先验框转换成中心，宽高的形式
#         #--------------------------------------------#
#         anchor_widths = anchor[:, 3] - anchor[:, 1]
#         anchor_heights = anchor[:, 2] - anchor[:, 0]
#         anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
#         anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights
#
#         regression_losses = []
#         classification_losses = []
#         for j in range(batch_size):
#             #-------------------------------------------------------#
#             #   取出每张图片对应的真实框、种类预测结果和回归预测结果
#             #-------------------------------------------------------#
#             bbox_annotation = annotations[j]
#             classification = classifications[j, :, :]
#             regression = regressions[j, :, :]
#
#             classification = torch.clamp(classification, 5e-4, 1.0 - 5e-4)
#
#             if len(bbox_annotation) == 0:
#                 #-------------------------------------------------------#
#                 #   当图片中不存在真实框的时候，所有特征点均为负样本
#                 #-------------------------------------------------------#
#                 alpha_factor = torch.ones_like(classification) * alpha
#                 alpha_factor = alpha_factor.type_as(classification)
#
#                 alpha_factor = 1. - alpha_factor
#                 focal_weight = classification
#                 focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
#
#                 #-------------------------------------------------------#
#                 #   计算特征点对应的交叉熵
#                 #-------------------------------------------------------#
#                 # bce = - (torch.log(1.0 - classification))
#                 #
#                 # cls_loss = focal_weight * bce
#                 #
#                 # classification_losses.append(cls_loss.sum())
#                 bce = - (torch.log(1.0 - classification))
#                 if self.class_weights is not None:
#                     self.class_weights = self.class_weights.to(classification.device)
#                     cls_loss = self.class_weights * focal_weight * bce
#                 else:
#                     cls_loss = focal_weight * bce
#
#                 classification_losses.append(cls_loss.sum())
#                 #-------------------------------------------------------#
#                 #   回归损失此时为0
#                 #-------------------------------------------------------#
#                 regression_losses.append(torch.tensor(0).type_as(classification))
#
#                 continue
#
#             #------------------------------------------------------#
#             #   计算真实框和先验框的交并比
#             #   targets                 num_anchors, num_classes
#             #   num_positive_anchors    正样本的数量
#             #   positive_indices        num_anchors,
#             #   assigned_annotations    num_anchors, 5
#             #------------------------------------------------------#
#             targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor,
#                                                                                         bbox_annotation, classification, cuda)
#
#             #------------------------------------------------------#
#             #   首先计算交叉熵loss
#             #------------------------------------------------------#
#             alpha_factor = torch.ones_like(targets) * alpha
#             alpha_factor = alpha_factor.type_as(classification)
#             #------------------------------------------------------#
#             #   这里使用的是Focal loss的思想，
#             #   易分类样本权值小
#             #   难分类样本权值大
#             #------------------------------------------------------#
#             alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
#             focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
#             focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
#
#             bce = - (targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
#             cls_loss = focal_weight * bce
#
#             #------------------------------------------------------#
#             #   把忽略的先验框的loss置为0
#             #------------------------------------------------------#
#             zeros = torch.zeros_like(cls_loss)
#             zeros = zeros.type_as(cls_loss)
#             cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)
#
#             classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
#
#             #------------------------------------------------------#
#             #   如果存在先验框为正样本的话
#             #------------------------------------------------------#
#             if positive_indices.sum() > 0:
#                 targets = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)
#                 #---------------------------------------------------#
#                 #   将网络应该有的预测结果和实际的预测结果进行比较
#                 #   计算smooth l1 loss
#                 #---------------------------------------------------#
#                 regression_diff = torch.abs(targets - regression[positive_indices, :])
#                 regression_loss = torch.where(
#                     torch.le(regression_diff, 1.0 / 9.0),
#                     0.5 * 9.0 * torch.pow(regression_diff, 2),
#                     regression_diff - 0.5 / 9.0
#                 )
#                 regression_losses.append(regression_loss.mean())
#             else:
#                 regression_losses.append(torch.tensor(0).type_as(classification))
#
#         # 计算平均loss并返回
#         c_loss = torch.stack(classification_losses).mean()
#         r_loss = torch.stack(regression_losses).mean()
#         loss = c_loss + r_loss
#         return loss, c_loss, r_loss
#
#
# # class FocalLoss(nn.Module):
# #     def __init__(self, class_weights=None):
# #         super(FocalLoss, self).__init__()
# #         self.class_weights = class_weights
# #
# #     def forward(self, classifications, regressions, anchors, annotations, alpha=0.25, gamma=2.0, cuda=True):
# #         batch_size = classifications.shape[0]
# #         dtype = regressions.dtype
# #         anchor = anchors[0, :, :].to(dtype)
# #
# #         anchor_widths = anchor[:, 3] - anchor[:, 1]
# #         anchor_heights = anchor[:, 2] - anchor[:, 0]
# #         anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
# #         anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights
# #
# #         regression_losses = []
# #         classification_losses = []
# #
# #         for j in range(batch_size):
# #             bbox_annotation = annotations[j]
# #             classification = classifications[j, :, :]
# #             regression = regressions[j, :, :]
# #
# #             classification = torch.clamp(classification, 5e-4, 1.0 - 5e-4)
# #
# #             if len(bbox_annotation) == 0:
# #                 # 如果没有标注框，则所有点都视为背景
# #                 alpha_factor = torch.ones_like(classification) * alpha
# #                 alpha_factor = 1. - alpha_factor
# #                 focal_weight = classification
# #                 focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
# #                 bce = - (torch.log(1.0 - classification))
# #                 classification_losses.append((focal_weight * bce).sum())
# #                 regression_losses.append(torch.tensor(0).type_as(classification))
# #                 continue
# #
# #             # 调用 get_target，返回 targets 和 positive_indices
# #             targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor, bbox_annotation,
# #                                                                                                classification, cuda)
# #             # 确保 positive_indices 和 regression 的大小一致
# #             positive_indices = positive_indices[:regression.shape[0]]  # 截断或填充 positive_indices 使其与 regression 匹配
# #
# #             if num_positive_anchors > 0:
# #                 # 编码目标框，用于回归损失计算
# #                 targets_encoded = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights,
# #                                               anchor_ctr_x, anchor_ctr_y)
# #
# #                 # 只保留与 positive_indices 对应的 targets_encoded 部分
# #                 targets_encoded = targets_encoded[:regression[positive_indices, :].shape[0]]
# #
# #                 # 计算回归损失
# #                 regression_diff = torch.abs(targets_encoded - regression[positive_indices, :])
# #                 regression_loss = torch.where(
# #                     torch.le(regression_diff, 1.0 / 9.0),
# #                     0.5 * 9.0 * torch.pow(regression_diff, 2),
# #                     regression_diff - 0.5 / 9.0
# #                 )
# #                 regression_losses.append(regression_loss.mean())
# #             else:
# #                 regression_losses.append(torch.tensor(0).type_as(classification))
# #
# #             # 计算分类损失
# #             alpha_factor = torch.ones_like(targets) * alpha
# #             alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
# #             focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
# #             focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
# #
# #             # 交叉熵损失
# #             bce = - (targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
# #             cls_loss = focal_weight * bce
# #
# #             # 忽略标签为 -1 的点
# #             cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))
# #
# #             classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
# #
# #         # 汇总损失
# #         c_loss = torch.stack(classification_losses).mean()
# #         r_loss = torch.stack(regression_losses).mean()
# #         total_loss = c_loss + r_loss
# #         return total_loss, c_loss, r_loss
#
#
# def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
#     def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
#         if iters <= warmup_total_iters:
#             # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
#             lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
#         elif iters >= total_iters - no_aug_iter:
#             lr = min_lr
#         else:
#             lr = min_lr + 0.5 * (lr - min_lr) * (
#                 1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
#             )
#         return lr
#
#     def step_lr(lr, decay_rate, step_size, iters):
#         if step_size < 1:
#             raise ValueError("step_size must above 1.")
#         n       = iters // step_size
#         out_lr  = lr * decay_rate ** n
#         return out_lr
#
#     if lr_decay_type == "cos":
#         warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
#         warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
#         no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
#         func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
#     else:
#         decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
#         step_size   = total_iters / step_num
#         func = partial(step_lr, lr, decay_rate, step_size)
#
#     return func
#
# def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
#     lr = lr_scheduler_func(epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
