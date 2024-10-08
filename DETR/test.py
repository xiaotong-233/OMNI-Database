import math
import os.path
import os
from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from hubconf import *
from util.misc import nested_tensor_from_tensor_list

torch.set_grad_enabled(False)

# 输入自己的数据集的类别
CLASSES = [
    '01', '02', '03', '04', '05', '06',
    '07', '08', '09', '10', '11', '12'
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, width, height):
    # img_w, img_h = size
    # b = box_cxcywh_to_xyxy(out_bbox)
    box_coords = box_cxcywh_to_xyxy(out_bbox)
    scale_tensor = torch.Tensor(
        [width, height, width, height]).to(
        torch.cuda.current_device()
    )
    return box_coords * scale_tensor


# def plot_results(pil_img, prob, boxes, image_item):
#     # 保存图片和labels文件夹的目录
#     save_dir = '/mnt/disk1/data0/jxt/classification/detr-main/newest_outputs'
#     # 保存画框后的图片目录
#     save_img_path = os.path.join(save_dir, image_item)
#     # labels文件夹目录
#     save_txt_dir = '/mnt/disk1/data0/jxt/dataset/data/frontandside/coco/test_annotated'
#
#     plt.figure(figsize=(16, 10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     colors = COLORS * 100
#     for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         cl = p.argmax()
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#
#         # 获取每个图片相应的txt文件的名字
#         # 如xxx.jpg，此处txt_id为xxx
#         txt_id = image_item[:-4]
#         # 获取每个txt的绝对路径 /home/exp1/xxx.txt
#         filename = os.path.join(save_txt_dir, f"{txt_id}.txt")
#         with open(filename, "a", encoding="utf-8") as f:
#             # 此处我只需要保存预测类别的序号和概率即可，
#             # 所以保存在txt文件里为cl-1即可，
#             # -1是因为我这里不需要N/A这个类别的序号
#             results = f"{cl - 1} {p[cl]} \n"
#             f.writelines(results)
#
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     # 保存画好的图片
#     plt.savefig(save_img_path, format='jpeg')
#     plt.close("all")
import os


def plot_results(pil_img, prob, boxes, image_item):
    save_dir = '/mnt/disk1/data0/jxt/classification/detr-main/newest_outputs/images'
    save_img_path = os.path.join(save_dir, image_item)
    save_txt_dir = '/mnt/disk1/data0/jxt/classification/detr-main/newest_outputs/labels'

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()

        # 打印cl值和相关数组的长度以进行调试
        print(f'Class index (cl): {cl}')
        print(f'Number of classes: {len(CLASSES)}')
        print(f'Probability vector length: {len(p)}')

        if cl < len(CLASSES) and cl < len(p):
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        else:
            text = 'Class index out of range!'  # 当cl值超出范围时显示错误信息

        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(save_img_path, format='jpeg')
    plt.close("all")


def detect(im, model, transform):
    device = torch.cuda.current_device()
    width = im.size[0]
    height = im.size[1]

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    img = img.to(device)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.25

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], width, height)
    return probas[keep], bboxes_scaled


if __name__ == "__main__":
    device = torch.device('cuda:0')
    # 这里与前面的num_classes数值相同，就是最大的category id值 + 1
    # 我有11类，故这里我填的12
    model = detr_resnet50(False, 13)
    # 自己已经训练好的权重 因为使用GPU所以map_location后面跟cuda
    state_dict = torch.load(r"/mnt/disk1/data0/jxt/classification/detr-main/newest_outputs/checkpoint0299.pth", map_location='cuda')
    model.load_state_dict(state_dict["model"])

    model.to(device)
    model.eval()

    # # 要预测的图片文件夹
    # image_file_path = os.listdir("/mnt/disk1/data0/jxt/dataset/data/frontandside/coco/test")
    # # 循环读入每一张图片
    # for image_item in image_file_path:
    #     image_path = os.path.join("/mnt/disk1/data0/jxt/dataset/data/frontandside/coco/test", image_item)
    #     im = Image.open(image_path)
    #     scores, boxes = detect(im, model, transform)
    #     plot_results(im, scores, boxes, image_item)

    # 设置数据集的根目录
    root_dir = "/mnt/disk1/data0/jxt/dataset/data/frontandside/coco/test"

    # 使用os.walk遍历所有子目录
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查文件扩展名，确保只处理图像文件
            if file.endswith(('.png', '.JPG', '.jpeg')):
                # 构建完整的文件路径
                image_path = os.path.join(subdir, file)
                im = Image.open(image_path)
                # 这里使用你的检测函数
                scores, boxes = detect(im, model, transform)
                # 这里使用你的绘图函数
                plot_results(im, scores, boxes, file)