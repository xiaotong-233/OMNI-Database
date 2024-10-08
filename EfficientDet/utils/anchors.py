import itertools

import numpy as np
import torch
import torch.nn as nn

#  用于生成anchor
class Anchors(nn.Module):
    def __init__(self, anchor_scale=2., pyramid_levels=[3, 4, 5, 6, 7]):   #anchor_scale锚框的尺度缩放因子，pyramid_levels金字塔的层级列表
        super().__init__()
        self.anchor_scale = anchor_scale
        self.pyramid_levels = pyramid_levels
        # strides步长为[8, 16, 32, 64, 128]， 特征点的间距
        self.strides = [2 ** x for x in self.pyramid_levels]   # 根据金字塔层级列表计算出对应的步长列表，表示特征图的缩放倍数。
        #尺度列表，表示不同层级的锚框相对于基准尺度的缩放比例
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 4.0), 2 ** (2.0 / 4.0)])
        # self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        #宽高比列表，表示不同层级的锚框的宽高比例
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    def forward(self, image):
        image_shape = image.shape[2:]
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        return anchor_boxes
