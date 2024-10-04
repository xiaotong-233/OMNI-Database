import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from .backbone import fasterrcnn
from .resnet50 import resnet50_with_rpn
from .graph import normalize_digraph
from .basic_block import *


# class GNN(nn.Module):
#     def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
#         super(GNN, self).__init__()
#         # in_channels: dim of node feature
#         # num_classes: num of nodes
#         # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
#         # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
#         # X' = ReLU(X + BN(V(X) + A x U(X)) )
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.relu = nn.ReLU()
#         self.metric = metric
#         self.neighbor_num = neighbor_num
#
#         # network
#         self.U = nn.Linear(self.in_channels, self.in_channels)
#         self.V = nn.Linear(self.in_channels, self.in_channels)
#         self.bnv = nn.BatchNorm1d(num_classes)
#
#         # init
#         self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
#         self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
#         self.bnv.weight.data.fill_(1)
#         self.bnv.bias.data.zero_()
#     def forward(self, x):
#         b, n, c = x.shape
#         # build dynamical graph
#         if self.metric == 'dots':
#             si = x.detach()
#             si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
#             threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
#             adj = (si >= threshold).float()
#         elif self.metric == 'cosine':
#             si = x.detach()
#             si = F.normalize(si, p=2, dim=-1)
#             si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
#             threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
#             adj = (si >= threshold).float()
#         elif self.metric == 'l1':
#             si = x.detach().repeat(1, n, 1).view(b, n, n, c)
#             si = torch.abs(si.transpose(1, 2) - si)
#             si = si.sum(dim=-1)
#             threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
#             adj = (si <= threshold).float()
#         else:
#             raise Exception("Error: wrong metric: ", self.metric)
#         # GNN process
#         A = normalize_digraph(adj)
#         aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
#         x = self.relu(x + self.bnv(aggregate + self.U(x)))
#         return x

# class Head_classes(nn.Module):
#     def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
#         super(Head_classes, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         class_linear_layers = []
#         for i in range(self.num_classes):
#             layer = LinearBlock(self.in_channels, self.in_channels)
#             class_linear_layers = class_linear_layers + [layer]
#         self.class_linear_layers = nn.ModuleList(class_linear_layers)
#         self.gnn_class = GNN(self.in_channels, self.num_classes, neighbor_num=int(neighbor_num), metric=metric)
#         self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
#         self.relu = nn.ReLU()
#         nn.init.xavier_uniform_(self.sc)
#     def forward(self, x):
#         # AFG
#         f_u_class = []
#         for layer in self.class_linear_layers:
#             f_u_class.append(layer(x).unsqueeze(1))
#         f_u_class = torch.cat(f_u_class, dim=1)
#         f_v_class = f_u_class.mean(dim=-2)
#
#         # FGG
#         f_v_class_gnn = self.gnn_class(f_v_class)
#         b, n, c = f_v_class_gnn.shape
#         sc = self.sc
#         sc = self.relu(sc)
#         sc = F.normalize(sc, p=2, dim=-1)
#         cl = F.normalize(f_v_class_gnn, p=2, dim=-1)
#         cl = (cl * sc.view(1, n, c)).sum(dim=-1)
#         return cl


# class GNN(nn.Module):
#     def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
#         super(GNN, self).__init__()
#         # in_channels: dim of node feature
#         # num_classes: num of nodes
#         # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
#         # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
#         # X' = ReLU(X + BN(V(X) + A x U(X)) )
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.relu = nn.ReLU()
#         self.metric = metric
#         self.neighbor_num = neighbor_num
#
#         # network
#         self.U = nn.Linear(516, 516)
#         self.V = nn.Linear(516, 516)
#         self.bnv = nn.BatchNorm1d(num_classes)
#
#         # init
#         self.U.weight.data.normal_(0, math.sqrt(2. / 516))
#         self.V.weight.data.normal_(0, math.sqrt(2. / 516))
#         self.bnv.weight.data.fill_(1)
#         self.bnv.bias.data.zero_()
#
#     def forward(self, f_v, node_features):
#         undated_f_v_list = []
#         node_features = node_features.unsqueeze(0)
#         combined_features = torch.cat((f_v.repeat(1, node_features.size(1), 1), node_features), dim=-1)
#         # combined_features = torch.cat((f_v.unsqueeze(1).repeat(1, node_features.size(1), 1), node_features), dim=-1)
#         b, n, c = combined_features.shape
#         # build dynamical graph
#         if self.metric == 'dots':
#             si = combined_features.detach()
#             si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
#             threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
#             adj = (si >= threshold).float()
#         elif self.metric == 'cosine':
#             si = combined_features.detach()
#             si = F.normalize(si, p=2, dim=-1)
#             si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
#             threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
#             adj = (si >= threshold).float()
#         elif self.metric == 'l1':
#             si = combined_features.detach().repeat(1, n, 1).view(b, n, n, c)
#             si = torch.abs(si.transpose(1, 2) - si)
#             si = si.sum(dim=-1)
#             threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
#             adj = (si <= threshold).float()
#         else:
#             raise Exception("Error: wrong metric: ", self.metric)
#         # GNN process
#         A = normalize_digraph(adj)
#         # 重塑 combined_features 以适应 self.V
#         combined_features_flat = combined_features.view(b * n, c)
#         V_output = self.V(combined_features_flat)
#         V_output_reshaped = V_output.view(b, n, -1)
#         aggregate = torch.einsum('b i j, b j k->b i k', A, V_output_reshaped)
#         combined_features = self.relu(combined_features + self.bnv(aggregate + self.U(combined_features)))
#         updated_f_v = combined_features[:, 0, :f_v.size(-1)]
#         return updated_f_v
# class Head_classes(nn.Module):
#     def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
#         super(Head_classes, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.neibor_num = neighbor_num
#         class_linear_layers = []
#         for i in range(self.num_classes):
#             layer = LinearBlock(self.in_channels, self.in_channels)
#             class_linear_layers += [layer]
#         self.class_linears = nn.ModuleList(class_linear_layers)
#         self.gnn = GNN(self.in_channels, self.num_classes, neighbor_num=neighbor_num, metric=metric)
#         self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
#         self.relu = nn.ReLU()
#         nn.init.xavier_uniform_(self.sc)
#
#     def forward(self, global_features, anchors):
#         # AFG
#         f_u = []
#         for i, layer in enumerate(self.class_linears):
#             transformed_features = layer(global_features).unsqueeze(1)
#             f_u.append(transformed_features)
#         f_u = torch.cat(f_u, dim=1)
#         f_v = f_u.mean(dim=-2)
#         node_features = anchors
#         # FGG
#         updated_f_v = self.gnn(f_v, node_features)
#         # b, c = updated_f_v.shape
#         sc = self.sc
#         sc = self.relu(sc)
#         sc = F.normalize(sc, p=2, dim=-1)
#         cl = F.normalize(updated_f_v, p=2, dim=-1)
#         sc = sc.view(1, sc.size(0), sc.size(1))
#         cl = cl.unsqueeze(1)
#         cl = (cl * sc).sum(dim=-1)
#         return cl, node_features

class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(in_channels, in_channels)
        self.V = nn.Linear(in_channels, in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, local_features):
        node_features = local_features.unsqueeze(0)
        # combined_features = torch.cat((f_v.repeat(1, node_features.size(1), 1), node_features), dim=-1)
        # combined_features = torch.cat((f_v.unsqueeze(1).repeat(1, node_features.size(1), 1), node_features), dim=-1)
        b, n, c = node_features.shape
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
        # 重塑 combined_features 以适应 self.V
        combined_features_flat = node_features.view(b * n, c)
        V_output = self.V(combined_features_flat)
        V_output_reshaped = V_output.view(b, n, -1)
        aggregate = torch.einsum('b i j, b j k->b i k', A, V_output_reshaped)
        combined_features = self.relu(node_features + self.bnv(aggregate + self.U(node_features)))
        updated_f_v = combined_features[:, 0, :]
        return updated_f_v
class Head_classes(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head_classes, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(in_channels, num_classes, neighbor_num=neighbor_num, metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(num_classes, in_channels)))
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.sc)

    def forward(self, local_features):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(local_features).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl
    # def forward(self, local_features):
    #     # 假设 anchors 是一个列表，其中每个元素代表一个样本的锚点特征
    #     # 我们可以使用 GNN 来更新每个样本的特征
    #     updated_features_list = []
    #     updated_features = self.gnn(local_features)
    #     updated_features_list.append(updated_features)
    #
    #     # 将更新后的特征列表转换为张量
    #     updated_features_tensor = torch.stack(updated_features_list, dim=0)
    #
    #     # 计算分类分数
    #     sc = self.relu(self.sc)
    #     sc = F.normalize(sc, p=2, dim=-1)
    #     cl = F.normalize(updated_features_tensor, p=2, dim=-1)
    #     sc = sc.view(1, sc.size(0), sc.size(1))
    #     cl = cl.unsqueeze(1)
    #     cl = (cl * sc).sum(dim=-1)
    #
    #     return cl, anchor

# class MEFARG(nn.Module):
#     def __init__(self, num_classes=32, neighbor_num=4, metric='dots'):
#         super(MEFARG, self).__init__()
#         self.backbone = fasterrcnn()
#         self.in_channels = 2048  # 通常ResNet最后一个卷积层的输出通道数
#         self.out_channels = self.in_channels // 4
#         self.global_linear_class = LinearBlock(self.in_channels, self.out_channels)
#         self.head_classes = Head_classes(self.out_channels, num_classes, neighbor_num=neighbor_num, metric=metric)
#
#     def forward(self, batch_images, batch_targets):
#         # batch_images: [(3,512,512), (3,512,512), ...]，每个元素是一个图像
#         # batch_targets: [targets1, targets2, ...]，每个元素是对应图像的目标信息
#         outputs = self.backbone(batch_images, batch_targets)
#         # 提取全局特征、局部特征和检测结果
#         global_features = outputs[1]
#         anchors = outputs[0]
#         # 将全局特征通过一个线性层
#         global_features = self.global_linear_class(global_features)  # （16， 512）
#         # 使用head_classes处理全局特征和检测结果
#         # cl, detection = self.head_classes(global_features, anchors)
#         cl_list = []
#         detection_list = []
#         for anchor in anchors:
#             cl, detection = self.head_classes(anchor)
#             cl_list.append(cl)
#             detection_list.append(detection)
#         return cl_list, detection_list

class MEFARG(nn.Module):
    def __init__(self, num_classes=32, neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        self.backbone = resnet50_with_rpn()
        # self.backbone = fasterrcnn()
        self.in_channels = 2048 * 7 * 7  # 通常ResNet最后一个卷积层的输出通道数
        self.out_channels = self.in_channels // 4 // 7 // 7
        self.global_linear_class = LinearBlock(self.in_channels, self.out_channels)
        self.head_classes = Head_classes(self.out_channels, num_classes, neighbor_num=neighbor_num, metric=metric)

    def forward(self, batch_images, batch_boxes):
        # 使用backbone进行特征提取
        results_list = []
        local_features_list = []
        processed_local_features = []
        for i, (image, boxes) in enumerate(zip(batch_images, batch_boxes)):
            image = image.unsqueeze(0)
            batch_index = torch.full((boxes.size(0), 1), i, dtype=boxes.dtype, device=boxes.device)
            proposals = torch.cat((batch_index, boxes), dim=1)
            output = self.backbone(image, proposals)
            local_feature = output['local_feature']
            local_features = self.global_linear_class(local_feature)
            cl, detection = self.head_classes(local_features)

        # results, global_features_list, local_features_list = self.backbone(batch_images, batch_targets)
        # global_features = torch.stack(global_features_list)  # 将全局特征列表堆叠成一个张量
        # global_features = self.global_linear_class(global_features)
        #     results_list.append(output['bbox_pred'])
        #     local_features_list.append(output['local_feature'])  # 使用 RoI Pooling 提取的局部特征
        # for local_feature in local_features_list:
        #     local_features = self.global_linear_class(local_feature)
        #     processed_local_features.append(local_features)
        return cl, detection

# class MEFARG(nn.Module):
#     def __init__(self, num_classes=32, neighbor_num=4, metric='dots'):
#         super(MEFARG, self).__init__()
#         # 使用faster_rcnn作为backbone
#         self.backbone = fasterrcnn()
#         self.in_channels = 2048  # 通常ResNet最后一个卷积层的输出通道数
#         self.out_channels = self.in_channels // 4
#         self.global_linear_class = LinearBlock(self.in_channels, self.out_channels)
#         self.head_classes = Head_classes(self.out_channels, num_classes, neighbor_num=neighbor_num, metric=metric)
#
#     def forward(self, x):  # x:(4,3,512,512), anchor_features:(batch_size, 填充最大值, 4)
#         # x: b d c
#         x = self.backbone(x)  # x:(4,256,2048)
#         # x = self.global_linear_class(x)  # x:(4,256,512)
#         cl, detection = self.head_classes(x)  # x:(4,256,512)
#         return cl, detection
