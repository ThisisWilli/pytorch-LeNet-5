'''
@project : LeNet-5
@author  : Hoodie_Willi
@description: $创建LeNet-5网络并训练
@time   : 2019-06-26 18:27:01
'''
import torch
from torch.utils.data import DataLoader
from data.Mydataset import MyDataset
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.autograd
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
from datetime import datetime
import cv2
import torchvision

train_txt_path = '../data/MNIST/txt/train.txt'
test_txt_path = '../data/MNIST/txt/test.txt'
valid_txt_path = '../data/MNIST/txt/valid.txt'
train_bs = 64
test_bs = 64

# 是否使用显卡
use_cuda = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # 不知道为什么这样，将3通道变为单通道
                                ])

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=transform)
test_data = MyDataset(txt_path=test_txt_path, transform=transform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=transform)
print(len(train_data), len(test_data))
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=test_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=True)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            # padding为2，保证输出大小仍为28*28，N = (W − F + 2P )/S+1 F:kernel size, S:stride
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 输出28*28*6
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出14*14*6
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 输出16*10*10
            nn.MaxPool2d(kernel_size=2, stride=2) # 输出16*5*5
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将tensor拉平
        x = x.reshape(-1, 16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = LeNet()
if use_cuda:
    model = model.cuda()
print(model)

# 选择优化方法
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 选择损失函数
ceriation = nn.CrossEntropyLoss()
max_epoch = 3

for epoch in range(max_epoch):
    running_loss = 0.0
    running_correct = 0
    print("Epoch  {}/{}".format(epoch, max_epoch))
    print("-" * 10)
    for data in train_loader:
        X_train, y_train = data
        # 有GPU加下面这行，没有不用加
        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_train, y_train = Variable(X_train), Variable(y_train)
        # print(y_train)
        outputs = model(X_train)
        # print(outputs)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = ceriation(outputs, y_train)

        loss.backward()
        optimizer.step()
        # running_loss += loss.data[0]
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0
    for data in test_loader:
        X_test, y_test = data
        # 有GPU加下面这行，没有不用加
        X_test, y_test = X_test.cuda(), y_test.cuda()
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)

    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(
        running_loss / len(train_data), 100 * running_correct / len(train_data) * 1.0,
        100 * testing_correct / len(test_data) * 1.0))

print('finish training!')
cost_time = (datetime.now() - now_time).seconds
print('cost time:{}s'.format(cost_time))


net_save_path = os.path.join('../model', 'net_params.pkl')
torch.save(model.state_dict(), net_save_path)


# 验证模型
for data in valid_loader:
    X_valid, y_valid = data
    X_valid, y_valid = X_valid.cuda(), y_valid.cuda()
    X_valid, y_valid = Variable(X_valid), Variable(y_valid)
    pred = model(X_valid)
    _, pred = torch.max(pred, 1)
    print("Predict Label is:", pred)
    print("Real Label is :", y_valid)



