import os
import xml.etree.ElementTree as ET

def change_class(xml_file_path, old_class_id, new_class_id):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        class_id = int(obj.find('name').text)
        if class_id == old_class_id:
            obj.find('name').text = str(new_class_id)

    tree.write(xml_file_path)

# 用法示例
data_dir = "/mnt/disk1/data0/jxt/dataset/data/tooth_numbering/Annotations"
old_class_id = 0
new_class_id = 29

# 遍历数据目录下的所有XML文件并修改类别
for filename in os.listdir(data_dir):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(data_dir, filename)
        change_class(xml_file_path, old_class_id, new_class_id)
