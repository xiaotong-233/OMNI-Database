import os
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.ops as ops
import torchvision.models
import torch.nn.functional as F
from torchvision.ops import nms
__all__ = ['resnet50']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

models_dir = os.path.expanduser('checkpoints')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # b,c,h,w = x.shape
        # x = x.view(b,c,-1).permute(0,2,1)
        return x

class RPN(nn.Module):
    def __init__(self, in_channels, anchor_sizes, anchor_ratios):
        super(RPN, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_score = nn.Conv2d(512, len(anchor_sizes) * len(anchor_ratios) * 2, kernel_size=1)  # 2是因为每个锚框有两个类别：前景和背景
        self.bbox_pred = nn.Conv2d(512, len(anchor_sizes) * len(anchor_ratios) * 4, kernel_size=1)  # 4是因为每个锚框需要4个坐标来表示
        # 注意：实际使用时，还需要添加锚框生成和处理逻辑

    def forward(self, x, feature_size=(16, 16), feature_stride=8):
        x = self.conv(x)
        x = self.relu(x)
        logits = self.cls_score(x)  #  (1,12,16,16)
        bbox_pred = self.bbox_pred(x)  #  (1,24,16,16)
        anchors = generate_anchor(self.anchor_sizes, self.anchor_ratios, feature_size, feature_stride)
        probs = torch.sigmoid(logits)  #  (1,12,16,16)
        foreground_scores = probs[:, 1::2, :, :]  #  (1,6,16,16)
        # 应用边界框调整
        refined_anchors = apply_box_deltas(anchors, bbox_pred)  #  (1536,4)
        # # 根据模型的状态设置 NMS 阈值
        # if self.training:
        #     score_threshold = 0.5
        #     iou_threshold = 0.3
        # else:
        #     score_threshold = 0.5
        #     iou_threshold = 0.3
        # 应用 NMS 来选择最终的提议框
        keep_indices = non_max_suppression(refined_anchors, foreground_scores, score_threshold=0.5, iou_threshold=0.3)
        proposals = anchors[keep_indices]
        return proposals, probs

class DetectionHead(nn.Module):
    def __init__(self, in_channels):
        super(DetectionHead, self).__init__()
        self.roi_pooler = ops.RoIAlign(output_size=(7, 7), spatial_scale=1 / 16, sampling_ratio=2)
        # self.box_regressor = nn.Linear(in_channels * 7 * 7, 4)
        self.box_regressor = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)  # 4个坐标值用于边界框回归
        )

    def forward(self, features, proposals):
        x = self.roi_pooler(features, proposals)  #  (16,2048,7,7)
        x = x.view(x.size(0), -1)
        bbox_deltas = self.box_regressor(x)  #  (16,2048,7,7)
        return bbox_deltas  # (16,4)

class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])
        self.rpn = RPN(2048, anchor_sizes=[64, 80, 128], anchor_ratios=[0.5, 1, 1.5])
        self.detection_head = DetectionHead(2048)
        self.global_feature_branch = nn.AdaptiveAvgPool2d((1, 1))  # 全局特征分支

    def forward(self, images):
        features = self.backbone(images)  # （1，2048，16，16）
        global_feature = self.global_feature_branch(features)  # (1,2048,1,1)
        global_feature = global_feature.view(global_feature.size(0), -1)  # 展平特征 (1,2048)
        # RPN
        proposals, _ = self.rpn(features)  # proposals:(16,4)  _:(1,12,16,16)
        # 添加批次索引到提议框
        batch_indices = torch.zeros((proposals.shape[0], 1), device=proposals.device)
        proposals_with_batch_indices = torch.cat((batch_indices, proposals), dim=1)  # (16,5)
        if features.is_cuda:
            proposals_with_batch_indices = proposals_with_batch_indices.cuda()
        bbox_deltas = self.detection_head(features, proposals_with_batch_indices)  # (16,4)
        refined_boxes = apply_box(proposals, bbox_deltas)  # (16,4)
        refined_boxes = torch.clamp(refined_boxes, min=0)
        refined_boxes = refined_boxes.to(batch_indices.device)
        refined_proposals_with_batch_indices = torch.cat((batch_indices, refined_boxes), dim=1)  # (16,5)
        if features.is_cuda:
            refined_proposals_with_batch_indices = refined_proposals_with_batch_indices.cuda()
        local_features = self.detection_head.roi_pooler(features, refined_proposals_with_batch_indices)  # (16,2048,7,7)
        local_features = local_features.view(local_features.size(0), -1)  # 展平特征  # (16,100352)
        return {'bbox_pred': refined_boxes, 'global_feature': global_feature, 'local_features': local_features}

def apply_box_deltas(anchors, bbox_pred, device='cuda:0'):
    """根据边界框回归调整锚框。"""
    anchors = anchors.to(device)
    bbox_pred = bbox_pred.to(device)
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    # 提取建议框的中心坐标和尺寸
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = bbox_pred[:, 0]
    dy = bbox_pred[:, 1]
    dw = bbox_pred[:, 2]
    dh = bbox_pred[:, 3]

    # 在 width、height、ctr_x 和 ctr_y 上增加一个维度以匹配 dx、dy、dw 和 dh 的形状
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(bbox_pred)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes

def apply_box(proposals, bbox_deltas):
    """根据边界框回归调整锚框。"""
    # 确保 proposals 和 bbox_deltas 在同一个设备上
    device = bbox_deltas.device
    proposals = proposals.to(device)
    # 提取建议框的中心坐标和尺寸
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    # 提取边界框回归的调整值
    dx = bbox_deltas[:, 0]
    dy = bbox_deltas[:, 1]
    dw = bbox_deltas[:, 2]
    dh = bbox_deltas[:, 3]

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

    # # 限制边界框坐标不超出图像边界
    # H, W = (512, 512)
    # pred_boxes[:, 0] = torch.clamp(pred_boxes[:, 0], min=0, max=W-1)
    # pred_boxes[:, 1] = torch.clamp(pred_boxes[:, 1], min=0, max=H-1)
    # pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], min=0, max=W-1)
    # pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], min=0, max=H-1)
    return pred_boxes

def non_max_suppression(boxes, scores, score_threshold=0.7, iou_threshold=0.4):
    """非最大抑制来去除重叠的边界框。"""
    scores = scores.reshape(-1)
    valid_indices = scores > score_threshold
    filtered_boxes = boxes[valid_indices]
    filtered_scores = scores[valid_indices]
    keep_indices = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold)  # 对每个批次的边界框进行 NMS
    return keep_indices

def generate_anchor(anchor_sizes=[48, 64, 80], anchor_ratios=[0.6, 1, 1.5], feature_size=(16, 16), feature_stride=8):
    anchors = []
    for size in anchor_sizes:
        for ratio in anchor_ratios:
            # 根据尺度和宽高比计算锚框的宽度和高度
            width = size * np.sqrt(ratio)
            height = size / np.sqrt(ratio)
            # 遍历特征图上的每个点，生成锚框
            for y in range(feature_size[0]):
                for x in range(feature_size[1]):
                    # 计算锚框在原始图像中的中心坐标
                    center_x = x * feature_stride + feature_stride // 2
                    center_y = y * feature_stride + feature_stride // 2
                    # 计算锚框的左上角和右下角坐标
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    # 将锚框坐标添加到列表中
                    anchors.append([x1, y1, x2, y2])
    # 将锚框列表转换为张量
    anchors = torch.tensor(anchors, dtype=torch.float32)
    return anchors
