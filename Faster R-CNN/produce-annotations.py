import os
import xml.etree.ElementTree as ET
from utils.utils import get_classes

classes_path        = 'model_data/classes.txt'
classes, _      = get_classes(classes_path)
# 数据集路径
dataset_path = "/mnt/disk1/data0/jxt/dataset/data/allsides_10"

# ImageSets文件夹路径
imagesets_path = os.path.join(dataset_path, "imageset1")

# Annotations文件夹路径
annotations_path = os.path.join(dataset_path, "GNN_annotated")

# JPEGImages文件夹路径
jpegimages_path = os.path.join(dataset_path, "JPEGImages")

# 读取train.txt文件
train_file = os.path.join(imagesets_path, "train.txt")
with open(train_file, "r") as f:
    train_folders = f.read().splitlines()

# 读取val.txt文件
val_file = os.path.join(imagesets_path, "val.txt")
with open(val_file, "r") as f:
    val_folders = f.read().splitlines()
#
# # 读取trainval.txt文件
# trainval_file = os.path.join(imagesets_path, "trainval.txt")
# with open(trainval_file, "r") as f:
#     trainval_folders = f.read().splitlines()

# 读取test.txt文件
test_file = os.path.join(imagesets_path, "test.txt")
with open(test_file, "r") as f:
    test_folders = f.read().splitlines()


# 生成保存数据路径和标注信息的txt文件
train_data_info_file = "number_train_data_10.txt"
val_data_info_file = "number_val_data_10.txt"
# trainval_data_info_file = "trainval_data_info.txt"
test_data_info_file = "number_test_data_10.txt"

# 遍历train.txt中的文件夹名
with open(train_data_info_file, "w") as f:
    for folder in train_folders:
        # 图像数据文件夹路径
        image_folder_path = os.path.join(jpegimages_path, folder)
        # 标注数据文件夹路径
        annotation_folder_path = os.path.join(annotations_path, folder)

        # 遍历标注数据文件夹中的二级子文件夹
        for subfolder in os.listdir(annotation_folder_path):
            subfolder_path = os.path.join(annotation_folder_path, subfolder)

            # 遍历二级子文件夹中的文件
            for file in os.listdir(subfolder_path):
                # 图像数据文件路径
                image_file_path = os.path.join(image_folder_path, subfolder, file)
                # 将文件名扩展名改为.JPG
                image_file_path = os.path.splitext(image_file_path)[0] + ".JPG"
                # 标注数据文件路径
                annotation_file_path = os.path.join(subfolder_path, file)

                # 解析标注数据文件，获取坐标和类别信息
                tree = ET.parse(annotation_file_path)
                # print(f"尝试解析的 XML 文件路径: {annotation_file_path}")
                root = tree.getroot()

                # 生成标注信息字符串
                annotation_info = ""
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    xmin = bbox.find("xmin").text
                    ymin = bbox.find("ymin").text
                    xmax = bbox.find("xmax").text
                    ymax = bbox.find("ymax").text
                    category = int(obj.find("name").text) - 1
                    annotation_info += f"{xmin},{ymin},{xmax},{ymax},{category} "

                # 去除末尾的逗号和空格
                annotation_info = annotation_info.rstrip(",")
                # # 打印图像路径和标注信息
                # data_info = f"{image_file_path} {annotation_info}"
                # print(data_info)

                # 将图像路径和标注信息写入txt文件
                data_info = f"{image_file_path} {annotation_info}\n"
                f.write(data_info)

# 处理val.txt
with open(val_data_info_file, "w") as f:
    for folder in val_folders:
        # 图像数据文件夹路径
        image_folder_path = os.path.join(jpegimages_path, folder)
        # 标注数据文件夹路径
        annotation_folder_path = os.path.join(annotations_path, folder)

        # 遍历标注数据文件夹中的二级子文件夹
        for subfolder in os.listdir(annotation_folder_path):
            subfolder_path = os.path.join(annotation_folder_path, subfolder)

            # 遍历二级子文件夹中的文件
            for file in os.listdir(subfolder_path):
                # 图像数据文件路径
                image_file_path = os.path.join(image_folder_path, subfolder, file)
                # 将文件名扩展名改为.JPG
                image_file_path = os.path.splitext(image_file_path)[0] + ".JPG"
                # 标注数据文件路径
                annotation_file_path = os.path.join(subfolder_path, file)

                # 解析标注数据文件，获取坐标和类别信息
                tree = ET.parse(annotation_file_path)
                # print(f"尝试解析的 XML 文件路径: {annotation_file_path}")
                root = tree.getroot()

                # 生成标注信息字符串
                annotation_info = ""
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    xmin = bbox.find("xmin").text
                    ymin = bbox.find("ymin").text
                    xmax = bbox.find("xmax").text
                    ymax = bbox.find("ymax").text
                    category = int(obj.find("name").text) - 1
                    annotation_info += f"{xmin},{ymin},{xmax},{ymax},{category} "

                # 去除末尾的逗号和空格
                annotation_info = annotation_info.rstrip(",")
                # # 打印图像路径和标注信息
                # data_info = f"{image_file_path} {annotation_info}"
                # print(data_info)

                # 将图像路径和标注信息写入txt文件
                data_info = f"{image_file_path} {annotation_info}\n"
                f.write(data_info)

# # 处理trainval.txt
# with open(trainval_data_info_file, "w") as f:
#     for folder in trainval_folders:
#         # 图像数据文件夹路径
#         image_folder_path = os.path.join(jpegimages_path, folder)
#         # 标注数据文件夹路径
#         annotation_folder_path = os.path.join(annotations_path, folder)
#
#         # 遍历标注数据文件夹中的二级子文件夹
#         for subfolder in os.listdir(annotation_folder_path):
#             subfolder_path = os.path.join(annotation_folder_path, subfolder)
#
#             # 遍历二级子文件夹中的文件
#             for file in os.listdir(subfolder_path):
#                 # 图像数据文件路径
#                 image_file_path = os.path.join(image_folder_path, subfolder, file)
#                 # 将文件名扩展名改为.JPG
#                 image_file_path = os.path.splitext(image_file_path)[0] + ".JPG"
#                 # 标注数据文件路径
#                 annotation_file_path = os.path.join(subfolder_path, file)
#
#                 # 解析标注数据文件，获取坐标和类别信息
#                 tree = ET.parse(annotation_file_path)
#                 root = tree.getroot()
#
#                 # 生成标注信息字符串
#                 annotation_info = ""
#                 for obj in root.findall("object"):
#                     bbox = obj.find("bndbox")
#                     xmin = bbox.find("xmin").text
#                     ymin = bbox.find("ymin").text
#                     xmax = bbox.find("xmax").text
#                     ymax = bbox.find("ymax").text
#                     category = int(obj.find("name").text)-1
#                     annotation_info += f"{xmin},{ymin},{xmax},{ymax},{category} "
#
#                 # 去除末尾的逗号和空格
#                 annotation_info = annotation_info.rstrip(",")
#                 # # 打印图像路径和标注信息
#                 # data_info = f"{image_file_path} {annotation_info}"
#                 # print(data_info)
#
#                 # 将图像路径和标注信息写入txt文件
#                 data_info = f"{image_file_path} {annotation_info}\n"
#                 f.write(data_info)

# 处理test.txt
with open(test_data_info_file, "w") as f:
    for folder in test_folders:
        # 图像数据文件夹路径
        image_folder_path = os.path.join(jpegimages_path, folder)
        # 标注数据文件夹路径
        annotation_folder_path = os.path.join(annotations_path, folder)

        # 遍历标注数据文件夹中的二级子文件夹
        for subfolder in os.listdir(annotation_folder_path):
            subfolder_path = os.path.join(annotation_folder_path, subfolder)

            # 遍历二级子文件夹中的文件
            for file in os.listdir(subfolder_path):
                # 图像数据文件路径
                image_file_path = os.path.join(image_folder_path, subfolder, file)
                # 将文件名扩展名改为.JPG
                image_file_path = os.path.splitext(image_file_path)[0] + ".JPG"
                # 标注数据文件路径
                annotation_file_path = os.path.join(subfolder_path, file)

                # 解析标注数据文件，获取坐标和类别信息
                tree = ET.parse(annotation_file_path)
                # print(f"尝试解析的 XML 文件路径: {annotation_file_path}")
                root = tree.getroot()

                # 生成标注信息字符串
                annotation_info = ""
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    xmin = bbox.find("xmin").text
                    ymin = bbox.find("ymin").text
                    xmax = bbox.find("xmax").text
                    ymax = bbox.find("ymax").text
                    category = str(obj.find("name").text)
                    annotation_info += f"{xmin},{ymin},{xmax},{ymax},{category} "

                # 去除末尾的逗号和空格
                annotation_info = annotation_info.rstrip(",")
                # # 打印图像路径和标注信息
                # data_info = f"{image_file_path} {annotation_info}"
                # print(data_info)

                # 将图像路径和标注信息写入txt文件
                data_info = f"{image_file_path} {annotation_info}\n"
                f.write(data_info)