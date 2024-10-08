import os
from PIL import Image
import xml.etree.ElementTree as ET

input_folder = "/mnt/disk1/data0/jxt/dataset/data/beifen"  # 输入文件夹路径
output_folder = "/mnt/disk1/data0/jxt/dataset/data/resize"  # 输出文件夹路径
# resize_ratio = 0.1  # 缩小比例
max_size = 512

input_image_folder = os.path.join(input_folder, "JPEGImages")
input_annotation_folder = os.path.join(input_folder, "Annotations")

output_image_folder = os.path.join(output_folder, "JPEGImages")
output_annotation_folder = os.path.join(output_folder, "Annotations")

# 创建输出文件夹
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_annotation_folder, exist_ok=True)


def resize_image(image_path):
    image = Image.open(image_path)
    width, height = image.size

    # 计算调整后的尺寸
    if width > height:
        new_width = max_size
        new_height = int((height / width) * max_size)
    else:
        new_width = int((width / height) * max_size)
        new_height = max_size

    resized_image = image.resize((new_width, new_height))
    return resized_image


def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".JPG"):
                # 处理图像文件
                image_path = os.path.join(root, file_name)
                image = Image.open(image_path)
                width, height = image.size  # 获取图像的宽度和高度

                resized_image = resize_image(image_path)
                # resized_image, new_width, new_height = resize_image(image_path)

                # 保存调整后的图像到输出文件夹的JPEGImages子文件夹
                output_subfolder = os.path.join(output_image_folder, os.path.relpath(root, input_image_folder))
                os.makedirs(output_subfolder, exist_ok=True)
                output_image_path = os.path.join(output_subfolder, file_name)
                resized_image.save(output_image_path)

                # 处理相应的标注数据的XML文件
                annotation_subfolder = os.path.join(input_annotation_folder, os.path.relpath(root, input_image_folder))
                xml_file_path = os.path.join(annotation_subfolder, file_name.replace(".JPG", ".xml"))
                if os.path.exists(xml_file_path):
                    tree = ET.parse(xml_file_path)
                    root_element = tree.getroot()  # 获取XML文档的根元素

                    # 获取调整后的图像尺寸
                    new_width, new_height = resized_image.size
                    # 更新标注数据中的宽度和高度为调整后图像的尺寸
                    size_element = root_element.find("size")
                    width_element = size_element.find("width")
                    height_element = size_element.find("height")
                    width_element.text = str(new_width)
                    height_element.text = str(new_height)

                    # 调整标注数据中的坐标
                    for obj in root_element.findall("object"):
                        bbox = obj.find("bndbox")
                        xmin = int(bbox.find("xmin").text)
                        ymin = int(bbox.find("ymin").text)
                        xmax = int(bbox.find("xmax").text)
                        ymax = int(bbox.find("ymax").text)

                        # 将坐标调整为相对于调整后的图像尺寸的比例
                        xmin_ratio = xmin / width
                        ymin_ratio = ymin / height
                        xmax_ratio = xmax / width
                        ymax_ratio = ymax / height

                        # 转换为调整后的图像尺寸下的坐标
                        new_xmin = int(new_width * xmin_ratio)
                        new_ymin = int(new_height * ymin_ratio)
                        new_xmax = int(new_width * xmax_ratio)
                        new_ymax = int(new_height * ymax_ratio)

                        # 更新XML中的坐标
                        bbox.find("xmin").text = str(new_xmin)
                        bbox.find("ymin").text = str(new_ymin)
                        bbox.find("xmax").text = str(new_xmax)
                        bbox.find("ymax").text = str(new_ymax)

                    output_annotation_subfolder = os.path.join(output_annotation_folder,
                                                               os.path.relpath(annotation_subfolder,
                                                                               input_annotation_folder))
                    os.makedirs(output_annotation_subfolder, exist_ok=True)
                    output_xml_path = os.path.join(output_annotation_subfolder, file_name.replace(".JPG", ".xml"))
                    tree.write(output_xml_path)

# 处理输入文件夹下的图像和标注数据
process_folder(input_image_folder)