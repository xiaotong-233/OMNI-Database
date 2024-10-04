# import numpy as np
import os
from scipy.sparse import lil_matrix

list_path = '../data/tooth/list'
# class_num = 12

# for i in range(1,4):
#     read_list_name = 'BP4D_train_label_fold'+str(i)+'.txt'
#     save_list_name = 'BP4D_train_AU_relation_fold'+str(i)+'.txt'
#     aus = np.loadtxt(os.path.join(list_path,read_list_name))
#     le = aus.shape[0]
#     new_aus = np.zeros((le, class_num * class_num))
#     for j in range(class_num):
#         for k in range(class_num):
#             new_aus[:,j*class_num+k] = 2 * aus[:,j] + aus[:,k]
#     np.savetxt(os.path.join(list_path,save_list_name),new_aus,fmt='%d')


read_list_name = 'train_label.txt'
save_list_name = 'tooth_train_relation.txt'
with open(os.path.join(list_path, read_list_name), 'r') as file:
    lines= file.readlines()
aus = [line.strip().split() for line in lines]
le = len(aus)
class_num = len(aus[0])
new_aus = [['0'] * (class_num * class_num) for _ in range(le)]
for i in range(le):
    for j in range(class_num):
        for k in range(class_num):
            # 确保新矩阵中的列表元素足够容纳每个计算结果
            while len(new_aus[i]) <= j * class_num + k:
                new_aus[i].append('0')
            # 直接将字符串拼接到新矩阵中
            new_aus[i][j * class_num + k] = aus[i][j] + aus[i][k]
with open(os.path.join(list_path, save_list_name), 'w') as file:
    for row in new_aus:
        file.write('\t'.join(map(str, row))+ '\n')