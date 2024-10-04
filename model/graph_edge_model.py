import torch
import torch.nn as nn
import math


class CrossAttn(nn.Module): # 交叉注意力机制，主要用于计算两个不同输入之间的注意力权重，并根据这些权重来合成新的特征输出。
    """ cross attention Module"""
    def __init__(self, in_channels):
        super(CrossAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5  # 用于缩放点积结果的缩放因子，以控制softmax函数前的数值稳定性。
        self.attend = nn.Softmax(dim=-1)  #用于计算注意力权重

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale  #  计算查询向量和键向量的点积，并乘以缩放因子。
        attn = self.attend(dots)  #  应用Softmax函数计算最终的注意力权重
        out = torch.matmul(attn, value)  #  使用注意力权重加权值向量，生成最终的输出特征。
        return out


# class GEM(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(GEM, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         # self.FAM, self.ARM: 两个 CrossAttn 实例，分别用于不同阶段的注意力计算。
#         self.FAM = CrossAttn(self.in_channels)
#         self.ARM = CrossAttn(self.in_channels)
#         self.edge_proj = nn.Linear(in_channels, in_channels)  #  一个线性变换，用于进一步处理注意力机制的输出。
#         self.bn = nn.BatchNorm2d(self.num_classes * self.num_classes)  #  批量归一化，用于规范化特征，加速训练过程，提高模型的稳定性。
#
#         self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
#         self.bn.weight.data.fill_(1)
#         self.bn.bias.data.zero_()
#
#     def forward(self, class_feature, global_feature):  # 输入的分类特征和全局特征
#         B, N, D, C = class_feature.shape
#         global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)  # 被重复并重排，以匹配 class_feature 的维度。
#         feat = self.FAM(class_feature, global_feature)  # 使用 self.FAM 处理 class_feature 和调整后的 global_feature。
#         # 分别重复 feat 以适配后续操作的维度需求。
#         feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
#         feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
#         feat = self.ARM(feat_start, feat_end)  # 使用 self.ARM 进行第二次注意力计算
#         edge = self.bn(self.edge_proj(feat))  # 应用线性变换和批量归一化
#         return edge

class GEM(nn.Module):
    def __init__(self, in_channels, num_classes, device='cuda:0'):
        super(GEM, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(in_channels)
        self.ARM = CrossAttn(in_channels)
        self.global_feature_transform = nn.Linear(512, in_channels)  # 调整全局特征维度
        self.edge_proj = nn.Linear(in_channels, in_channels)  # Assume in_channels is the dimension of each feature
        # self.bn = nn.BatchNorm2d(num_classes * num_classes)  # Adjust dimensions according to what is needed

        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        # self.bn.weight.data.fill_(1)
        # self.bn.bias.data.zero_()
    def forward(self, class_feature, global_feature):
        class_feature = class_feature.to(self.device)
        global_feature = global_feature.to(self.device)

        global_feature = self.global_feature_transform(global_feature)
        class_feature = class_feature.unsqueeze(0)
        num_anchors = class_feature.size(1)
        global_feature = global_feature.unsqueeze(0)
        B, N, D, C = class_feature.shape  # class_feature shape is (batch_size, num_samples_per_image, feature_dim)
        expanded_global_feature = global_feature.repeat(1, N, 1).view(B, N, -1, C)  # A simple repeat and reshape to align dimensions

        # First attention mechanism
        feat = self.FAM(class_feature, expanded_global_feature)
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end)

        bn = nn.BatchNorm2d(num_anchors * num_anchors).to(feat.device)
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        edge = bn(self.edge_proj(feat))
        return edge

    # def forward(self, class_feature, global_feature):
    #     global_feature = self.global_feature_transform(global_feature)
    #
    #     B, N, C = class_feature.shape  # class_feature shape is (batch_size, num_samples_per_image, feature_dim)
    #     _, D = global_feature.shape  # global_features shape is (batch_size, num_global_features, feature_dim)
    #
    #     # Adjust global features to match the dimensions of class_feature
    #     # Repeat each global feature N times to match the number of class_features
    #     expanded_global_feature = global_feature.repeat(N, 1).view(N, -1, C)  # A simple repeat and reshape to align dimensions
    #
    #     # First attention mechanism
    #     feat = self.FAM(class_feature.reshape(-1, C), expanded_global_feature.reshape(-1, C))
    #
    #     # Prepare for the second attention mechanism
    #     feat_end = feat.repeat(1, N).view(B, N * N, C)
    #     feat_start = feat.repeat(N, 1).view(B, N * N, C)
    #
    #     # Second attention mechanism
    #     feat = self.ARM(feat_start, feat_end)
    #
    #     # Linear transformation and batch normalization
    #     edge = self.bn(self.edge_proj(feat).view(B, N * N, -1))
    #     return edge


