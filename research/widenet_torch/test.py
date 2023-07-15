'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-15 14:11:28
'''
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import configs 
import parsers as ps 
import widenet

args = ps.get_parser()

print(args)
# 设置随机数种子
configs.setup_seed(args.seed)

   
config = configs.CONFIGS[args.model_type]
model = widenet.WideNet(config)
# model = widenet.PatchEncoder(4,768,0.1)

# 假设输入张量 x 的形状为 (batch_size, channel, width, height)
batch_size = 10
channel = 3
width = 32
height = 32


# 创建随机输入张量 x
input = torch.randn(batch_size, channel, width, height)

output = model(input)
print(input.shape)
print(output.shape)