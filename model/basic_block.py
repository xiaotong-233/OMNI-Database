import torch
import torch.nn as nn
import math

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.bn(x)
        x = self.relu(x).permute(0, 2, 1)
        return x





# class LinearBlock(nn.Module):
#     def __init__(self, in_features, out_features=None, drop=0.0):
#         super().__init__()
#         out_features = out_features or in_features
#         self.fc = nn.Linear(in_features, out_features)
#         self.bn = nn.BatchNorm2d(out_features)
#         self.relu = nn.ReLU(inplace=True)
#         self.drop = nn.Dropout(drop)
#         self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
#         self.bn.weight.data.fill_(1)
#         self.bn.bias.data.zero_()
#
#     def forward(self, x):
#         x = self.drop(x)
#         x = self.fc(x)
#         x = x.unsqueeze(2).unsqueeze(3)  # 添加两个维度以匹配BatchNorm2d的输入要求
#         x = self.bn(x)
#         x = self.relu(x)
#         x = x.squeeze(3).squeeze(2)  # 移除添加的维度
#         return x