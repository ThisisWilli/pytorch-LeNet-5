'''
@project : LeNet-5
@author  : Hoodie_Willi
@description: ${}
@time   : 2019-06-26 18:27:01
'''
import torch
from torch.utils.data import DataLoader
from data.Mydataset import MyDataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.autograd
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import cv2
train_txt_path = '../data/MNIST/txt/train.txt'
test_txt_path = '../data/MNIST/txt/test.txt'
train_bs = 64
test_bs = 64

# use cuda or not
use_cuda = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
                                ])
# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=transform)
test_data = MyDataset(txt_path=test_txt_path, transform=transform)
print(len(train_data), len(test_data))
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=test_bs, shuffle=True)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = LeNet()
if use_cuda:
    model = model.cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
ceriation = nn.CrossEntropyLoss()
# for epoch in range(3):
#     # trainning
#     ave_loss = 0
#
#     for batch_idx, (x, target) in enumerate(train_loader):
#         running_correct = 0
#         optimizer.zero_grad()
#         if use_cuda:
#             x, target = x.cuda(), target.cuda()
#         x, target = Variable(x), Variable(target)
#         out = model(x)
#
#         _, pred = torch.max(out.data, 1)
#         loss = ceriation(out, target)
#         ave_loss = ave_loss * 0.9 + loss.item() * 0.1
#         loss.backward()
#         optimizer.step()
#         running_correct += torch.sum(pred == target.data)
#         if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
#             print(
#             '==>>> epoch: {}, batch index: {}, train loss: {:.6f}, train_acc: {:.6f}'.format(
#                 epoch, batch_idx + 1, ave_loss, 10*running_correct / 100 *1.0))
#     # testing
#     correct_cnt, ave_loss = 0, 0
#     total_cnt = 0
#     for batch_idx, (x, target) in enumerate(test_loader):
#         if use_cuda:
#             x, target = x.cuda(), target.cuda()
#         x, target = Variable(x), Variable(target)
#         out = model(x)
#         loss = ceriation(out, target)
#         _, pred_label = torch.max(out.data, 1)
#         #total_cnt += len(x)
#         total_cnt += x.data.size().item()
#         # total_cnt += x.data.size()[0]
#         correct_cnt += (pred_label == target.data).sum()
#         # smooth average
#         ave_loss = ave_loss * 0.9 + loss.data * 0.1
#
#         if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
#             print(
#             '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
#                 epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))
for epoch in range(3):
    running_loss = 0.0
    running_correct = 0
    print("Epoch  {}/{}".format(epoch, 3))
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
        # print("ok")
        # print("**************%s"%running_corrrect)

    print("train ok ")
    testing_correct = 0
    for data in test_loader:
        X_test, y_test = data
        # 有GPU加下面这行，没有不用加
        X_test, y_test = X_test.cuda(), y_test.cuda()
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
        # print(testing_correct)

    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(
        running_loss / len(train_data), 100 * running_correct / len(train_data),
        100 * testing_correct / len(test_data)))

