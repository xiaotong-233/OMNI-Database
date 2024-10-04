import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from .backbone import fasterrcnn
from .resnet50 import FasterRCNN
from .graph import normalize_digraph
from .basic_block import *

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

    def forward(self, local_features, bbox_pred):
        # AFG
        device = local_features.device
        num_anchors = local_features.size(0)
        class_linear_layers = nn.ModuleList([LinearBlock(self.in_channels, self.in_channels).to(device) for _ in range(num_anchors)])
        # class_linear_layers = class_linear_layers.to(device)  # Move the entire ModuleList to the device
        # class_linear_layers = [LinearBlock(self.in_channels, self.in_channels).to(device) for _ in range(num_anchors)]
        # class_linears = nn.ModuleList(class_linear_layers)
        f_u = [layer(local_features).unsqueeze(1) for layer in class_linear_layers]
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
        return cl_probs, bbox_pred
#
# GNN模型使用已调好的主函数（不要动）
class MEFARG(nn.Module):
    def __init__(self, num_classes=18, neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        self.backbone = FasterRCNN()
        self.in_channels = 2048 * 7 * 7  # 通常ResNet最后一个卷积层的输出通道数
        self.out_channels = self.in_channels // 4 // 7 // 7
        self.global_linear_class = LinearBlock(self.in_channels, self.out_channels)
        self.head_classes = Head_classes(self.out_channels, num_classes, neighbor_num=neighbor_num, metric=metric)

    def forward(self, batch_images, batch_boxes):
        # 使用backbone进行特征提取
        class_list = []
        detection_list = []
        local_features_list = []
        processed_local_features = []
        for i, (image, boxes) in enumerate(zip(batch_images, batch_boxes)):
            image = image.unsqueeze(0)
            output = self.backbone(image)
            bbox_pred = output['bbox_pred']
            local_feature = output['local_features']
            local_features = self.global_linear_class(local_feature)  # (16,512)
            cl, detection = self.head_classes(local_features, bbox_pred)
            class_list.append(cl)
            detection_list.append(detection)
        # results, global_features_list, local_features_list = self.backbone(batch_images, batch_targets)
        # global_features = torch.stack(global_features_list)  # 将全局特征列表堆叠成一个张量
        # global_features = self.global_linear_class(global_features)

        #     local_features_list.append(output['local_feature'])  # 使用 RoI Pooling 提取的局部特征
        # for local_feature in local_features_list:
        #     local_features = self.global_linear_class(local_feature)
        #     processed_local_features.append(local_features)
        return class_list, detection_list
