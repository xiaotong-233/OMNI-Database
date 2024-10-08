import torch
pretrained_weights  = torch.load('detr-r50-e632da11.pth')

#NWPU数据集，10类
num_class = 11    #类别数+1，1为背景
pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1, 256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)
torch.save(pretrained_weights, "detr-r50_%d.pth"%num_class)
