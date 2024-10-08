import datetime
import os

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import mplcursors

import shutil
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import decodebox, non_max_suppression
from .utils_map import get_coco_map, get_map


class LossHistory():
    # 初始化类的属性
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    # 用于向损失列表中添加新的损失值，并将其写入文件和日志记录中
    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    # 用于生成并保存损失值的可视化图表
    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()  #创建新图表
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            # 通过scipy.signal.savgol_filter对损失值进行平滑处理，减少噪声
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    # 修改！！！！
    # def loss_plot(self):
    #     iters = range(len(self.losses))
    #
    #     plt.figure(figsize=(10, 6))  # Increase figure size for better visibility
    #
    #     plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
    #     plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
    #
    #     try:
    #         if len(self.losses) < 25:
    #             num = 5
    #         else:
    #             num = 15
    #         # Use scipy.signal.savgol_filter to smooth the losses
    #         plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smooth train loss')
    #         plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
    #     except:
    #         pass
    #
    #     plt.grid(True)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend(loc="upper right")
    #
    #     # Mark key points, e.g., 9th epoch and 20th epoch
    #     plt.scatter([9, 20], [self.losses[8], self.losses[19]], color='blue', marker='o')
    #     plt.scatter([9, 20], [self.val_loss[8], self.val_loss[19]], color='green', marker='o')
    #
    #     # Annotate the points with epoch and loss values
    #     plt.annotate(f'Epoch 9\nTrain Loss: {self.losses[8]:.2f}\nVal Loss: {self.val_loss[8]:.2f}',
    #                  xy=(9, self.losses[8]), xytext=(12, self.losses[8] + 5),
    #                  arrowprops=dict(facecolor='blue', arrowstyle='->'))
    #
    #     plt.annotate(f'Epoch 20\nTrain Loss: {self.losses[19]:.2f}\nVal Loss: {self.val_loss[19]:.2f}',
    #                  xy=(20, self.losses[19]), xytext=(24, self.losses[19] - 5),
    #                  arrowprops=dict(facecolor='blue', arrowstyle='->'))
    #
    #     # Set xlim to zoom in after the 20th epoch
    #     plt.xlim(20, len(self.losses))
    #
    #     plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
    #
    #     plt.cla()
    #     plt.close("all")



class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence  #置信度阈值
        self.nms_iou            = nms_iou   #非极大值抑制阈值
        self.letterbox_image    = letterbox_image   #是否使用灰条进行图像缩放
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    # 用于处理单张图像的目标检测结果
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)   #np.transpose是转置操作

        with torch.no_grad():  # 禁用梯度计算
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   传入网络当中进行预测
            #---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)
            
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            anchors = anchors[:, :regression.shape[1], :]
            outputs     = decodebox(regression, anchors, self.input_shape)  #解码，将模型的输出转换为真实世界坐标
            # 使用非极大抑制，保留最相关的结果
            results     = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape, 
                                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
               
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                image       = Image.open(line[0])
                #------------------------------#
                #   获得预测框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+ image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
