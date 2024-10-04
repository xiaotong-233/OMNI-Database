import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
import math
from .backbone import fasterrcnn
from .resnet50 import FasterRCNN
from .resnet import resnet18, resnet34, resnet50, resnet101
from .graph import normalize_digraph
from .basic_block import *
# from torchvision.models import resnet50

class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(in_channels, in_channels)
        self.V = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, local_features):
        node_features = local_features.unsqueeze(0)  # （1，13，512）
        b, n, c = node_features.shape
        self.bnv = nn.BatchNorm1d(n).to(local_features.device)
        # build dynamical graph
        if self.metric == 'dots':
            si = node_features.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()
        elif self.metric == 'cosine':
            si = node_features.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()
        elif self.metric == 'l1':
            si = node_features.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()
        else:
            raise Exception("Error: wrong metric: ", self.metric)
        # GNN process
        A = normalize_digraph(adj)
        local_features_flat = local_features.view(b * n, c)  # Reshape local_features to be two-dimensional
        V_output = self.V(local_features_flat)  # Apply the linear layer
        V_output_reshaped = V_output.view(b, n, -1)  # Reshape back to three dimensions
        aggregate = torch.einsum('b i j, b j k->b i k', A, V_output_reshaped)
        node_features = self.relu(node_features + self.bnv(aggregate + self.U(node_features)))
        # class_scores_flat = self.fc(node_features.view(b * n, c))
        # class_scores = class_scores_flat.view(b, n, -1)
        class_scores = self.fc(node_features.squeeze(0))
        return class_scores
class Head_classes(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head_classes, self).__init__()
        self.in_channels = in_channels
        self.num_classes =num_classes
        self.linear_block = LinearBlock(self.in_channels, self.in_channels)
        self.gnn = GNN(in_channels, num_classes, neighbor_num=neighbor_num, metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(num_classes, in_channels)))
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.sc)
        # 创建一个ModuleList，其中包含max_anchors个LinearBlock实例
        # self.class_linears = nn.ModuleList([LinearBlock(self.in_channels, self.in_channels) for _ in range(max_anchors)])

    def forward(self, local_feature):
        # AFG
        device = local_feature.device
        num_anchors = local_feature.size(0)
        class_linear_layers = nn.ModuleList([LinearBlock(self.in_channels, self.in_channels).to(device) for _ in range(num_anchors)])
        # class_linear_layers = class_linear_layers.to(device)  # Move the entire ModuleList to the device
        # class_linear_layers = [LinearBlock(self.in_channels, self.in_channels).to(device) for _ in range(num_anchors)]
        # class_linears = nn.ModuleList(class_linear_layers)
        f_u = [layer(local_feature).unsqueeze(1) for layer in class_linear_layers]
        f_u = torch.cat(f_u, dim=1)  # （13，13，512）
        f_v = f_u.mean(dim=-2)  #（13，512）

        # FGG：提供一个向量作为节点特征来描述每个节点的激活以及它与其他节点的关联
        f_v = self.gnn(f_v)
        n, c = f_v.shape
        # 通过全连接层将特征维度转换为 512
        fc = nn.Linear(c, 512).to(device)
        cl = fc(f_v)
        sc = self.sc.to(device)
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        sc = sc.view(1, 18, 512)  # 重塑 sc 以匹配 cl 的形状
        cl = cl.view(n, 1, 512)  # 重塑 cl 以进行逐元素乘法
        cl = (cl * sc).sum(dim=-1)  # 对特征进行加权求和
        # cl = F.normalize(f_v, p=2, dim=-1)
        # cl = torch.matmul(cl, sc.T)  # 结果形状为 [1, n, 18]
        cl_probs = torch.softmax(cl, dim=-1)
        # classes = torch.argmax(cl_probs, dim=-1)
        return cl_probs


# # GNN模型使用已调好的主函数（不要动）
# class MEFARG(nn.Module):
#     def __init__(self, num_classes=18, neighbor_num=4, metric='dots'):
#         super(MEFARG, self).__init__()
#         self.in_channels =1024  # 通常ResNet最后一个卷积层的输出通道数
#         self.out_channels = 512
#         self.global_linear_class = LinearBlock(self.in_channels, self.out_channels)
#         self.head_classes = Head_classes(self.out_channels, num_classes, neighbor_num=neighbor_num, metric=metric)
#
#     def forward(self, batch_images, batch_boxes, local_features):
#         # 使用backbone进行特征提取
#         class_list = []
#         for i, (image, boxes, local_feature) in enumerate(zip(batch_images, batch_boxes, local_features)):
#             local_feature = local_feature[0]
#             local_feature = local_feature[0]
#             local_feature = self.global_linear_class(local_feature)
#             cl = self.head_classes(local_feature)
#             class_list.append(cl)
#
#         return class_list

class FeatureFusionModule(nn.Module):
    def __init__(self, global_feat_dim, local_feat_dim, output_dim):
        super(FeatureFusionModule, self).__init__()
        # 初始化全连接层以融合处理后的全局和局部特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(global_feat_dim + local_feat_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, global_feat, local_feat):
        # 全局平均池化
        global_pooled = F.adaptive_avg_pool2d(global_feat, (1, 1)).view(global_feat.size(0), -1)
        local_pooled = F.adaptive_avg_pool2d(local_feat, (1, 1)).view(local_feat.size(0), -1)

        # 确保全局特征扩展到与局部特征匹配的批次大小
        if global_pooled.size(0) != local_pooled.size(0):
            expanded_global_feat = global_pooled.repeat(local_pooled.size(0), 1)
        else:
            expanded_global_feat = global_pooled

        # 拼接全局和局部特征
        combined_feat = torch.cat([expanded_global_feat, local_pooled], dim=1)

        # 使用定义的全连接层来融合特征
        fused_feat = self.fusion_layer(combined_feat)
        return fused_feat

# GNN模型使用已调好的主函数（不要动）
class MEFARG(nn.Module):
    def __init__(self, num_classes=18, neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        self.net_cropped_features = resnet50()
        self.net_initial_features = resnet34()
        self.net_cropped_features = nn.Sequential(*list(self.net_cropped_features.children())[:-2])
        self.net_initial_features = nn.Sequential(*list(self.net_initial_features.children())[:-2])
        self.in_channels = 2048
        self.out_channels = self.in_channels // 4
        self.fusion_module = FeatureFusionModule(2048, 2048, 2048)
        self.global_linear_class = LinearBlock(self.in_channels, self.out_channels)
        self.head_classes = Head_classes(self.out_channels, num_classes, neighbor_num=neighbor_num, metric=metric)

    # def forward(self, batch_images, cropped_images):
    #     # 使用backbone进行特征提取
    #     class_list = []
    #     for i, (image, cropped_image) in enumerate(zip(batch_images, cropped_images)):
    #         image = image.unsqueeze(0)
    #         cropped_image_tensor = torch.cat(cropped_image, dim=0)
    #         cropped_features = self.net_cropped_features(cropped_image_tensor)
    #         initial_features = self.net_initial_features(image)
    #         fused_features = self.fusion_module(initial_features, cropped_features)
    #         combined_features = self.global_linear_class(fused_features)
    #         cl = self.head_classes(combined_features)
    #         class_list.append(cl)
    #     return class_list

    def forward(self, batch_images, cropped_images):
        # 处理initial images
        initial_features = self.net_initial_features(batch_images)

        # 合并并处理所有cropped images
        all_cropped_images = torch.cat([torch.cat(crops, dim=0) for crops in cropped_images], dim=0)
        cropped_features = self.net_cropped_features(all_cropped_images)
        # fused_features = self.fusion_module(initial_features, cropped_features)
        # combined_features = self.global_linear_class(fused_features)
        # cl = self.head_classes(combined_features)
        # return cl
        # 重构用于特征融合的initial_features以匹配cropped_features的数量
        initial_features_expanded = []
        for i, crops in enumerate(cropped_images):
            num_crops = len(crops)
            initial_features_expanded.extend([initial_features[i]] * num_crops)
        initial_features_expanded = torch.cat(initial_features_expanded, dim=0)

        # 特征融合
        fused_features = self.fusion_module(initial_features_expanded, cropped_features)
        combined_features = self.global_linear_class(fused_features)

        # 分类
        class_outputs = self.head_classes(combined_features)

        # 重新组织class_outputs以匹配原始的batch_images结构
        class_list = []
        start_idx = 0
        for crops in cropped_images:
            num_crops = len(crops)
            end_idx = start_idx + num_crops
            class_list.append(class_outputs[start_idx:end_idx])
            start_idx = end_idx

        return class_list

