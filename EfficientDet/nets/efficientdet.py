import torch
import torch.nn as nn
from utils.anchors import Anchors

from nets.efficientnet import EfficientNet as EffNet
from nets.layers import (Conv2dStaticSamePadding, MaxPool2dStaticSamePadding,
                         MemoryEfficientSwish, Swish)


#----------------------------------#
#   Xception中深度可分离卷积
#   先3x3的深度可分离卷积
#   再1x1的普通卷积
#----------------------------------#
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            # 获取到了efficientnet的最后三层，对其进行通道的下压缩
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # 对输入进来的p5进行宽高的下采样
            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            # BIFPN第一轮的时候，跳线那里并不是同一个in
            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # 简易注意力机制的weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """ bifpn模块结构示意图
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        #------------------------------------------------#
        #   当phi=1、2、3、4、5的时候使用fast_attention
        #   获得三个shape的有效特征层
        #   分别是C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        #------------------------------------------------#
        if self.first_time:
            #------------------------------------------------------------------------#
            #   第一次BIFPN需要 下采样 与 调整通道 获得 p3_in p4_in p5_in p6_in p7_in
            #------------------------------------------------------------------------#
            p3, p4, p5 = inputs
            #-------------------------------------------#
            #   首先对通道数进行调整
            #   C3 64, 64, 40 -> 64, 64, 64
            #-------------------------------------------#
            p3_in = self.p3_down_channel(p3)

            #-------------------------------------------#
            #   首先对通道数进行调整
            #   C4 32, 32, 112 -> 32, 32, 64
            #                  -> 32, 32, 64
            #-------------------------------------------#
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            #-------------------------------------------#
            #   首先对通道数进行调整
            #   C5 16, 16, 320 -> 16, 16, 64
            #                  -> 16, 16, 64
            #-------------------------------------------#
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)

            #-------------------------------------------#
            #   对C5进行下采样，调整通道数与宽高
            #   C5 16, 16, 320 -> 8, 8, 64
            #-------------------------------------------#
            p6_in = self.p5_to_p6(p5)
            #-------------------------------------------#
            #   对P6_in进行下采样，调整宽高
            #   P6_in 8, 8, 64 -> 4, 4, 64
            #-------------------------------------------#
            p7_in = self.p6_to_p7(p6_in)

            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td= self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td= self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td= self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in_2还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td+ weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in_2还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td+ weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td+ weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td= self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td= self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td= self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td+ weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td+ weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td+ weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        # 当phi=6、7的时候使用_forward
        if self.first_time:
            # 第一次BIFPN需要下采样与降通道获得
            # p3_in p4_in p5_in p6_in p7_in
            p3, p4, p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p6_td= self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td= self.conv5_up(self.swish(p5_in_1 + self.p5_upsample(p6_td)))

            p4_td= self.conv4_up(self.swish(p4_in_1 + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in_2 + p4_td+ self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in_2 + p5_td+ self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td+ self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_td= self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td= self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_td)))

            p4_td= self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td+ self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in + p5_td+ self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td+ self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class BoxNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(BoxNet, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        # 每一个有效特征层对应的Batchnor不同
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(5)])
        # 9
        # 4 中心，宽高
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        # 对每个特征层循环
        for feat, bn_list in zip(inputs, self.bn_list):
            # 每个特征层需要进行num_layer次卷积+标准化+激活函数
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)
        # 进行一个堆叠
        feats = torch.cat(feats, dim=1)

        return feats

class ClassNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(ClassNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list  = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        # 每一个有效特征层对应的BatchNorm2d不同
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(5)])
        # num_anchors = 9
        # num_anchors num_classes
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        # 对每个特征层循环
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                # 每个特征层需要进行num_layer次卷积+标准化+激活函数
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)
        # 进行一个堆叠
        feats = torch.cat(feats, dim=1)
        # 取sigmoid表示概率
        feats = feats.sigmoid()

        return feats

class EfficientNet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', pretrained)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            #------------------------------------------------------#
            #   取出对应的特征层，如果某个EffcientBlock的步长为2的话
            #   意味着它的前一个特征层为有效特征层
            #   除此之外，最后一个EffcientBlock的输出为有效特征层
            #------------------------------------------------------#
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes = 18, phi = 0, pretrained = False):
        super(EfficientDetBackbone, self).__init__()
        #--------------------------------#
        #   phi指的是efficientdet的版本
        #--------------------------------#
        self.phi = phi
        #---------------------------------------------------#
        #   backbone_phi指的是该efficientdet对应的efficient
        #---------------------------------------------------#
        self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]
        #--------------------------------#
        #   BiFPN所用的通道数
        #--------------------------------#
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        #--------------------------------#
        #   BiFPN的重复次数
        #--------------------------------#
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        #---------------------------------------------------#
        #   Effcient Head卷积重复次数
        #---------------------------------------------------#
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        #---------------------------------------------------#
        #   基础的先验框大小
        #---------------------------------------------------#
        #修改（已改回）
        self.anchor_scale = [2., 4., 4., 4., 4., 4., 4., 5.]
        num_anchors = 9

        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        #------------------------------------------------------#
        #   在经过多次BiFPN模块的堆叠后，我们获得的fpn_features
        #   假设我们使用的是efficientdet-D0包括五个有效特征层：
        #   P3_out      64,64,64
        #   P4_out      32,32,64
        #   P5_out      16,16,64
        #   P6_out      8,8,64
        #   P7_out      4,4,64
        #------------------------------------------------------#
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.phi],
                    conv_channel_coef[phi],
                    True if _ == 0 else False,
                    attention=True if phi < 6 else False)
              for _ in range(self.fpn_cell_repeats[phi])])

        self.num_classes = num_classes
        #------------------------------------------------------#
        #   创建efficient head
        #   可以将特征层转换成预测结果
        #------------------------------------------------------#
        self.regressor      = BoxNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                    num_layers=self.box_class_repeats[self.phi])

        self.classifier     = ClassNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                    num_classes=num_classes, num_layers=self.box_class_repeats[self.phi])

        self.anchors        = Anchors(anchor_scale=self.anchor_scale[phi])

        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        #-------------------------------------------#
        self.backbone_net   = EfficientNet(self.backbone_phi[phi], pretrained)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs)

        return features, regression, classification, anchors



# import torch
# import torch.nn as nn
# from utils.anchors import Anchors
#
# from nets.efficientnet import EfficientNet as EffNet
# from nets.layers import (Conv2dStaticSamePadding, MemoryEfficientSwish, Swish)
#
#
# class SeparableConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
#         super(SeparableConvBlock, self).__init__()
#         if out_channels is None:
#             out_channels = in_channels
#
#         self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1,
#                                                       groups=in_channels, bias=False)
#         self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
#
#         self.norm = norm
#         if self.norm:
#             self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
#
#         self.activation = activation
#         if self.activation:
#             self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
#
#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         x = self.pointwise_conv(x)
#
#         if self.norm:
#             x = self.bn(x)
#
#         if self.activation:
#             x = self.swish(x)
#
#         return x
#
#
# class BoxNet(nn.Module):
#     def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
#         super(BoxNet, self).__init__()
#         self.num_layers = num_layers
#
#         self.conv_list = nn.ModuleList(
#             [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_layers)])
#         self.bn_list = nn.ModuleList(
#             [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for _ in range(num_layers)]) for _ in
#              range(3)])  # Adjust for 3 scales
#         self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
#         self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
#
#     def forward(self, inputs):
#         feats = []
#         for feat, bn_list in zip(inputs, self.bn_list):
#             for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
#                 feat = conv(feat)
#                 feat = bn(feat)
#                 feat = self.swish(feat)
#             feat = self.header(feat)
#
#             feat = feat.permute(0, 2, 3, 1)
#             feat = feat.contiguous().view(feat.shape[0], -1, 4)
#             feats.append(feat)
#         feats = torch.cat(feats, dim=1)
#         return feats
#
#
# class ClassNet(nn.Module):
#     def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
#         super(ClassNet, self).__init__()
#         self.num_anchors = num_anchors
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.conv_list = nn.ModuleList(
#             [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_layers)])
#         self.bn_list = nn.ModuleList(
#             [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for _ in range(num_layers)]) for _ in
#              range(3)])  # Adjust for 3 scales
#         self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
#         self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
#
#     def forward(self, inputs):
#         feats = []
#         for feat, bn_list in zip(inputs, self.bn_list):
#             for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
#                 feat = conv(feat)
#                 feat = bn(feat)
#                 feat = self.swish(feat)
#             feat = self.header(feat)
#
#             feat = feat.permute(0, 2, 3, 1)
#             feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
#                                           self.num_classes)
#             feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)
#
#             feats.append(feat)
#         feats = torch.cat(feats, dim=1)
#         feats = feats.sigmoid()
#         return feats
#
#
# class EfficientNet(nn.Module):
#     def __init__(self, phi, pretrained=False):
#         super(EfficientNet, self).__init__()
#         model = EffNet.from_pretrained(f'efficientnet-b{phi}', pretrained)
#         del model._conv_head
#         del model._bn1
#         del model._avg_pooling
#         del model._dropout
#         del model._fc
#         self.model = model
#
#     def forward(self, x):
#         x = self.model._conv_stem(x)
#         x = self.model._bn0(x)
#         x = self.model._swish(x)
#         feature_maps = []
#
#         last_x = None
#         for idx, block in enumerate(self.model._blocks):
#             drop_connect_rate = self.model._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self.model._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if block._depthwise_conv.stride == [2, 2]:
#                 feature_maps.append(last_x)
#             elif idx == len(self.model._blocks) - 1:
#                 feature_maps.append(x)
#             last_x = x
#         del last_x
#         return feature_maps[1:]
#
#
# class EfficientDetBackbone(nn.Module):
#     def __init__(self, num_classes=18, phi=0, pretrained=False):
#         super(EfficientDetBackbone, self).__init__()
#         self.phi = phi
#         self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]
#         self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]  # BoxNet 和 ClassNet 期望的通道数
#         self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
#         self.anchor_scale = [2., 4., 4., 4., 4., 4., 4., 5.]
#         num_anchors = 9
#
#         conv_channel_coef = {
#             0: [40, 112, 320],
#             1: [40, 112, 320],
#             2: [48, 120, 352],
#             3: [48, 136, 384],
#             4: [56, 160, 448],
#             5: [64, 176, 512],
#             6: [72, 200, 576],
#             7: [72, 200, 576],
#         }
#
#         self.num_classes = num_classes
#
#         # 添加 1x1 卷积层以调整特征图通道数
#         self.adjust_p3 = nn.Conv2d(in_channels=conv_channel_coef[phi][0], out_channels=self.fpn_num_filters[self.phi],
#                                    kernel_size=1)
#         self.adjust_p4 = nn.Conv2d(in_channels=conv_channel_coef[phi][1], out_channels=self.fpn_num_filters[self.phi],
#                                    kernel_size=1)
#         self.adjust_p5 = nn.Conv2d(in_channels=conv_channel_coef[phi][2], out_channels=self.fpn_num_filters[self.phi],
#                                    kernel_size=1)
#         # Optionally apply upsampling to match the original feature map sizes
#         self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#
#         self.regressor = BoxNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
#                                 num_layers=self.box_class_repeats[self.phi])
#         self.classifier = ClassNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
#                                    num_classes=num_classes, num_layers=self.box_class_repeats[self.phi])
#         self.anchors = Anchors(anchor_scale=self.anchor_scale[phi])
#
#         self.backbone_net = EfficientNet(self.backbone_phi[phi], pretrained)
#
#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()
#
#     def forward(self, inputs):
#         # 从 EfficientNet Backbone 获取特征图
#         _, p3, p4, p5 = self.backbone_net(inputs)
#
#         # 调整特征图的通道数
#         p3 = self.adjust_p3(p3)
#         p4 = self.adjust_p4(p4)
#         p5 = self.adjust_p5(p5)
#
#         features = (p3, p4, p5)
#
#         regression = self.regressor(features)
#         classification = self.classifier(features)
#         anchors = self.anchors(inputs)
#
#         return features, regression, classification, anchors
