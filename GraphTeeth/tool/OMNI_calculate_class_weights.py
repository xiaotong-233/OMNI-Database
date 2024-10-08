import numpy as np
def calculate_class_weights(input_file, output_file):
    try:

        with open(input_file, 'r') as infile:
            lines = infile.readlines()


        class_frequencies = np.zeros(10)
        total_samples = 0

        for line in lines:

            num_classes = int(line.strip().split(',')[-1])  
            if 1 <= num_classes <= 10:
                class_frequencies[num_classes - 1] += 1
                total_samples += 1


        class_rates = class_frequencies / total_samples

        class_weights = 1.0 / class_rates

        class_weights = class_weights / class_weights.sum() * 10


        with open(output_file, 'w') as outfile:
            for class_num, weight in enumerate(class_weights, start=1):
                outfile.write(f"{weight:.6f}\t")

        print(f"Class weights calculated and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")



input_txt = 'annotation_train_label.txt'  # 将这里替换为你的包含类别数的txt文件路径
output_txt = 'train_weights_all10.txt'  # 输出文件路径

calculate_class_weights(input_txt, output_txt)


