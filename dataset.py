import numpy as np
import random
from PIL import Image
from util.utils import *
import os
import torch
import cv2
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence


# def make_dataset(image_list, label_list, au_relation=None):
#     len_ = len(image_list)
#     if au_relation is not None:
#         images = [(image_list[i].strip(),  label_list[i], au_relation[i]) for i in range(len_)]
#     else:
#         images = [(image_list[i].strip(),  label_list[i]) for i in range(len_)]
#     return images
# def make_dataset(image_list, label_list):
#     len_ = len(image_list)
#     images = [(image_list[i].strip(),  label_list[i]) for i in range(len_)]
#     return images

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class BP4D(Dataset):
    def __init__(self, annotation_lines, train=True, val=False, loader=default_loader):
    # def __init__(self, root_path, annotation_lines, train=True, val=False, stage=1, loader=default_loader):
        self.annotation_lines = annotation_lines
        self._train = train
        self._val = val
        # self._transform = transform
        self.loader = loader
        self.input_shape = [512, 512]
        self.length = len(annotation_lines)

    def __len__(self):
        # return len(self.data_list)
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # box指的是图像数据对应的真实框
        image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self._train)

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float16)), (2, 0, 1))
        box = np.array(box, dtype=np.float16)
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

        # get_random_data方法用于对图像进行随机的缩放和扭曲，并进行数据增强。
        # 首先，读取图像并转换为RGB格式。
        # 获取图像的大小和目标框。
        # 如果random为False，按比例缩放图像并在图像周围添加灰条，同时调整真实框的坐标。
        # 如果random为True，对图像进行随机缩放和扭曲，然后根据随机参数调整图像色域，并对真实框进行坐标调整。
        # 最后，返回处理后的图像和真实框。

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 如果图像高度小于512，则在上下添加灰度条
        if ih < h:
            padding = (h - ih) // 2
            new_image = Image.new('RGB', (iw, h), (128, 128, 128))
            new_image.paste(image, (0, padding))
            image = new_image
            # 更新标注框坐标
            if len(box) > 0:
                box[:, [1, 3]] += padding

        image_data = np.array(image, np.uint8)

        # # 对图像进行色域变换的逻辑保持不变
        # r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # # ---------------------------------#
        # #   将图像转到HSV上
        # # ---------------------------------#
        # hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        # dtype = image_data.dtype
        # # ---------------------------------#
        # #   应用变换
        # # ---------------------------------#
        # x = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        #
        # image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * iw / iw
            box[:, [1, 3]] = box[:, [1, 3]] * ih / ih
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box


def GNN_collect_fn(batch):
    images = []
    label_tensors = []
    for img, boxes in batch:
        images.append(torch.tensor(img))
        anchor_boxes = []
        class_labels = []
        for box in boxes:
            anchor_box = torch.tensor(box[:4])  # 提取anchor信息
            class_label = torch.tensor(box[4])  # 提取class信息
            anchor_boxes.append(anchor_box)
            class_labels.append(class_label)
        anchor_boxes = torch.stack(anchor_boxes)
        class_labels = torch.stack(class_labels)
        label_tensor = {'boxes': anchor_boxes, 'classes': class_labels}
        label_tensors.append(label_tensor)
    images = torch.stack(images)
    return images, label_tensors
