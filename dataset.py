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



def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class OMNI(Dataset):
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
