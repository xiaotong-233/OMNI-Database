import os
import numpy as np
import pandas as pd


# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

list_path_prefix = '../data/tooth/list/'

# def get_AUlabels(seq, task, path):
# 	path_label = os.path.join(path, '{sequence}_{task}.csv'.format(sequence=seq, task=task))
# 	usecols = ['0', '1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']
# 	df = pd.read_csv(path_label, header=0, index_col=0, usecols=usecols)
# 	frames = [str(item) for item in list(df.index.values)]
# 	frames_path = ['{}/{}/{}'.format(seq, task, item) for item in frames]
# 	labels = df.values
# 	# 返回的frames是list，值是排好序的int变量，指示对应的帧。labels是N*12的np.ndarray，对应AU标签
# 	return frames_path, labels

train_annotation_path = '/mnt/disk1/data0/jxt/ME-GraphAU-main/data/tooth/list/tooth_test_img_path_fold1.txt'
val_annotation_path = '/mnt/disk1/data0/jxt/ME-GraphAU-main/data/tooth/list/tooth_test_img_path_fold2.txt'
test_annotation_path = '/mnt/disk1/data0/jxt/ME-GraphAU-main/data/tooth/list/tooth_test_img_path_fold3.txt'
train_lable_path = '/mnt/disk1/data0/jxt/ME-GraphAU-main/data/tooth/list/tooth_test_label_fold1.txt'
val_lable_path = '/mnt/disk1/data0/jxt/ME-GraphAU-main/data/tooth/list/tooth_test_label_fold2.txt'
test_lable_path = '/mnt/disk1/data0/jxt/ME-GraphAU-main/data/tooth/list/tooth_test_label_fold3.txt'

####################################################################################################
# with open(list_path_prefix + 'tooth_test_img_path_fold3.txt','w') as f:
#     u = 0
frames = None
labels = None
len_f = 0
with open(test_annotation_path) as f:
 	test_lines = f.readlines()
with open(test_lable_path) as f:
	test_lables = f.readlines()
if frames is None:
	labels = test_lables
	frames = test_lines  # str list
else:
	labels = np.concatenate((labels, test_lables), axis=0)  # np.ndarray
	frames = frames + test_lines  # str list
tooth_image_path_list_part1 = frames
tooth_image_label_part1 = labels
# 将每个帧的图像名写入txt
# for frame in tooth_image_path_list_part1:
# 	frame_img_name = frame + '.JPG'
# 	with open(list_path_prefix + 'tooth_test_img_path_fold3.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# # 将标签保存到'BP4D_test_label_fold3.txt'
# np.savetxt(list_path_prefix + 'tooth_test_label_fold3.txt', tooth_image_label_part1 ,fmt='%d', delimiter=' ')


####################################################################################################
# with open(list_path_prefix + 'tooth_test_img_path_fold2.txt','w') as f:
#     u = 0
frames = None
labels = None
len_f = 0
with open(val_annotation_path) as f:
 	val_lines = f.readlines()
with open(val_lable_path) as f:
	val_lables = f.readlines()
if frames is None:
	labels = val_lables
	frames = val_lines  # str list
else:
	labels = np.concatenate((labels, val_lables), axis=0)  # np.ndarray
	frames = frames + val_lines  # str list
tooth_image_path_list_part2 = frames
tooth_image_label_part2 = labels
#
# for frame in tooth_image_path_list_part2:
# 	frame_img_name = frame + '.JPG'
# 	with open(list_path_prefix + 'tooth_test_img_path_fold2.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# np.savetxt(list_path_prefix + 'tooth_test_label_fold2.txt', tooth_image_label_part2, fmt='%d', delimiter=' ')

####################################################################################################
# with open(list_path_prefix + 'tooth_test_img_path_fold1.txt','w') as f:
#     u = 0
frames = None
labels = None
len_f = 0
with open(train_annotation_path) as f:
 	train_lines = f.readlines()
with open(train_lable_path) as f:
	train_lables = f.readlines()
if frames is None:
	labels = train_lables
	frames = train_lines  # str list
else:
	labels = np.concatenate((labels, train_lables), axis=0)  # np.ndarray
	frames = frames + train_lines  # str list
tooth_image_path_list_part3 = frames
tooth_image_label_part3 = labels
#
# for frame in tooth_image_path_list_part3:
# 	frame_img_name = frame + '.JPG'
# 	with open(list_path_prefix + 'tooth_test_img_path_fold1.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# np.savetxt(list_path_prefix + 'tooth_test_label_fold1.txt', tooth_image_label_part3, fmt='%d', delimiter=' ')


#################################################################################
train_img_label_fold1_list = tooth_image_path_list_part1 + tooth_image_path_list_part2

with open(list_path_prefix + 'tooth_train_img_path_fold1.txt', 'w') as f:
    for frame in train_img_label_fold1_list:
        frame_img_name = frame
        f.write(frame_img_name)  # 添加换行符以分隔路径

train_img_label_fold1_numpy = np.concatenate((tooth_image_label_part1, tooth_image_label_part2), axis=0)
np.savetxt(list_path_prefix + 'tooth_train_label_fold1.txt', train_img_label_fold1_numpy, fmt='%d')
# with open(list_path_prefix + 'tooth_train_img_path_fold1.txt','w') as f:
#     u = 0
# train_img_label_fold1_list = tooth_image_path_list_part1 + tooth_image_path_list_part2
# for frame in train_img_label_fold1_list:
# 	frame_img_name = frame
# 	with open(list_path_prefix + 'tooth_train_img_path_fold1.txt', 'a+') as f:
# 		f.write(frame_img_name)
# train_img_label_fold1_numpy = np.concatenate((tooth_image_label_part1, tooth_image_label_part2), axis=0)
# np.savetxt(list_path_prefix + 'tooth_train_label_fold1.txt', train_img_label_fold1_numpy, fmt='%d')

#################################################################################
with open(list_path_prefix + 'tooth_train_img_path_fold2.txt','w') as f:
    u = 0
train_img_label_fold2_list = tooth_image_path_list_part1 + tooth_image_path_list_part3
for frame in train_img_label_fold2_list:
	frame_img_name = frame
	with open(list_path_prefix + 'tooth_train_img_path_fold2.txt', 'a+') as f:
		f.write(frame_img_name)
train_img_label_fold2_numpy = np.concatenate((tooth_image_label_part1, tooth_image_label_part3), axis=0)
np.savetxt(list_path_prefix + 'tooth_train_label_fold2.txt', train_img_label_fold2_numpy, fmt='%d')

#################################################################################
with open(list_path_prefix + 'tooth_train_img_path_fold3.txt','w') as f:
    u = 0
train_img_label_fold3_list = tooth_image_path_list_part2 + tooth_image_path_list_part3
for frame in train_img_label_fold3_list:
	frame_img_name = frame
	with open(list_path_prefix + 'tooth_train_img_path_fold3.txt', 'a+') as f:
		f.write(frame_img_name)
train_img_label_fold3_numpy = np.concatenate((tooth_image_label_part2, tooth_image_label_part3), axis=0)
np.savetxt(list_path_prefix + 'tooth_train_label_fold3.txt', train_img_label_fold3_numpy, fmt='%d')
