import numpy as np
def calculate_class_weights(input_file, output_file):
    try:
        # 读取输入文件
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        # 计算每一类的频率
        class_frequencies = np.zeros(10)
        total_samples = 0

        for line in lines:
            # 每行的最后一个元素是类别编号
            num_classes = int(line.strip().split(',')[-1])  # 用逗号分隔并取最后一个元素
            if 1 <= num_classes <= 10:
                class_frequencies[num_classes - 1] += 1
                total_samples += 1

        # 计算每个类别的频率
        class_rates = class_frequencies / total_samples

        # 计算每个类别的权重
        class_weights = 1.0 / class_rates
        # class_weights /= class_weights.sum()
        class_weights = class_weights / class_weights.sum() * 10

        # 写入输出文件
        with open(output_file, 'w') as outfile:
            for class_num, weight in enumerate(class_weights, start=1):
                outfile.write(f"{weight:.6f}\t")

        print(f"Class weights calculated and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


# 用法示例
input_txt = '/mnt/disk1/data0/jxt/dataset/data/allsides_10/list/annotation_train_label.txt'  # 将这里替换为你的包含类别数的txt文件路径
output_txt = '/mnt/disk1/data0/jxt/dataset/data/allsides_10/list/train_weights_all10.txt'  # 输出文件路径

calculate_class_weights(input_txt, output_txt)


# list_path_prefix = '/mnt/disk1/data0/jxt/ME-GraphAU-main/data/tooth/list/'
#
#
# def process_line(line):
#     words = line.split()
#     values = [float(word) for word in words if word.replace('.', '', 1).isdigit()]  # 过滤掉不能转换为浮点数的部分
#     return values
#
# with open(list_path_prefix + 'train_label.txt', 'r') as file:
#     # 逐行读取文件并处理每行
#     lines = file.readlines()
#     imgs_AUoccur = [process_line(line) for line in lines]
#
#     # 找到所有行中最大的列数
#     max_columns = max(len(row) for row in imgs_AUoccur)
#
#     # 对于较短的行，在末尾填充零
#     imgs_AUoccur = [row + [0] * (max_columns - len(row)) for row in imgs_AUoccur]
#
#     # 计算每个类别的AU出现率
#     AUoccur_rate = []
#     for i in range(len(imgs_AUoccur[0])):
#         class_column = [row[i] for row in imgs_AUoccur]
#         class_occurrence_rate = sum(1 for value in class_column if value > 0) / float(len(class_column))
#         AUoccur_rate.append(class_occurrence_rate)
#
#     # 计算每个类别的AU权重
#     AU_weight = [1.0 / rate if rate > 0 else 0 for rate in AUoccur_rate]
#     AU_weight_sum = sum(AU_weight)
#     AU_weight = [round(weight / AU_weight_sum, 6) for weight in AU_weight]
#
#     # 保存每个类别的AU权重到文件
#     with open(list_path_prefix + 'train_weight.txt', 'w') as weight_file:
#         weight_file.write('\t'.join(map(str, AU_weight)))