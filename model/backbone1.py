import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import nms, box_iou, batched_nms, clip_boxes_to_image, RoIAlign
import numpy as np
import torchvision.ops as ops
import torchvision.models


class RPN(nn.Module):
    def __init__(self, in_channels, anchor_sizes, anchor_ratios):
        super(RPN, self).__init__()
        self.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_ratios)
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_score = nn.Conv2d(512, self.anchor_generator.num_anchors_per_location()[0] * 2, kernel_size=1)
        self.bbox_pred = nn.Conv2d(512, self.anchor_generator.num_anchors_per_location()[0] * 4, kernel_size=1)
        # 注意：实际使用时，还需要添加锚框生成和处理逻辑

    def forward(self, features, image_size):
        feature = features[0]
        x = self.conv(feature)
        x = self.relu(x)
        logits = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        anchors = self.anchor_generator([feature], [image_size])[0]
        probs = torch.sigmoid(logits)  #  (1,12,16,16)
        foreground_scores = probs[:, 1::2, :, :].reshape(-1)  #  (1,6,16,16)
        proposals = apply_box(anchors, bbox_pred.view(-1, 4))  # Apply bbox deltas
        proposals = clip_boxes_to_image(proposals, image_size)  # Clip boxes to image
        keep_indices = batched_nms(proposals, foreground_scores, torch.zeros_like(foreground_scores), iou_threshold=0.3)
        return proposals[keep_indices], probs

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.roi_pooler = RoIAlign(output_size=(7, 7), spatial_scale=1 / 16, sampling_ratio=2)
        self.box_regressor = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)  # 4个坐标值用于边界框回归
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)  # 类别预测
        )

    def forward(self, features, proposals):
        x = self.roi_pooler(features, proposals)  #  (16,2048,7,7)
        x = x.view(x.size(0), -1)
        bbox_deltas = self.box_regressor(x)  #  (16,2048,7,7)
        class_logits = self.classifier(x)
        return bbox_deltas, class_logits  # (16,4)

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        out_channels = self.backbone.out_channels  # FPN 的输出通道数
        self.rpn = RPN(out_channels, anchor_sizes=[64, 80, 128], anchor_ratios=[0.5, 1, 1.5])
        self.detection_head = DetectionHead(out_channels, num_classes)
        self.global_feature_branch = nn.AdaptiveAvgPool2d((1, 1))  # 全局特征分支

    def forward(self, images):
        features = self.backbone(images)['0']
        global_feature = self.global_feature_branch(features)
        global_feature = global_feature.view(global_feature.size(0), -1)
        image_size = (images.shape[-2], images.shape[-1])  # (height, width)
        # RPN
        proposals, _ = self.rpn([features], image_size)  # proposals:(16,4)  _:(1,12,16,16)
        # 添加批次索引到提议框
        batch_indices = torch.zeros((proposals.shape[0], 1), device=proposals.device)
        proposals_with_batch_indices = torch.cat((batch_indices, proposals), dim=1)  # (16,5)
        if features.is_cuda:
            proposals_with_batch_indices = proposals_with_batch_indices.cuda()
        bbox_deltas, class_logits = self.detection_head([features], proposals_with_batch_indices)  # (16,4)
        refined_boxes = apply_box(proposals, bbox_deltas)  # (16,4)
        refined_boxes = torch.clamp(refined_boxes, min=0)
        refined_boxes = refined_boxes.to(batch_indices.device)
        refined_proposals_with_batch_indices = torch.cat((batch_indices, refined_boxes), dim=1)  # (16,5)
        if features.is_cuda:
            refined_proposals_with_batch_indices = refined_proposals_with_batch_indices.cuda()
        local_features = self.detection_head.roi_pooler(features, refined_proposals_with_batch_indices)  # (16,2048,7,7)
        local_features = local_features.view(local_features.size(0), -1)  # 展平特征  # (16,100352)
        return {'bbox_pred': refined_boxes, 'global_feature': global_feature, 'local_features': local_features}

def apply_box(anchors, bbox_pred):
    """根据边界框回归调整锚框。"""
    # 确保 proposals 和 bbox_deltas 在同一个设备上
    device = bbox_pred.device
    proposals = anchors.to(device)
    # 提取建议框的中心坐标和尺寸
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    # 提取边界框回归的调整值
    dx = bbox_pred[:, 0]
    dy = bbox_pred[:, 1]
    dw = bbox_pred[:, 2]
    dh = bbox_pred[:, 3]

    # 计算调整后的边界框的中心和尺寸
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # 计算调整后的边界框的坐标
    pred_boxes = torch.zeros_like(proposals)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

