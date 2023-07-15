import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import configs 
import parsers as ps 
from resnet import *
import widenet

args = ps.get_parser()

print(args)

# 基础设置
device = 'cuda'
num_epochs = args.epochs
batch_size = args.batch_size  
lr = args.base_lr 

# 文件读存路径
data_dir = args.data_dir
ckt_dir = args.checkpoint_dir

# 设置随机数种子
configs.setup_seed(args.seed)

# 图像预处理
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 数据集下载
train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                            train=False, 
                                            transform=transforms.ToTensor())

# 数据载入
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

    
# model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
config = configs.CONFIGS[args.model_type]
model = widenet.WideNet(config).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 更新学习率
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 训练数据集
total_step = len(train_loader)
curr_lr = lr
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # 延迟学习率
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# 测试网络模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# S将模型保存
torch.save(model.state_dict(), ckt_dir + 'resnet.ckpt')

