'''
@project : LeNet-5
@author  : Hoodie_Willi
@description: ${}
@time   : 2019-06-26 19:12:49
'''

import torch
from PIL import Image
# file_name = 'D:\\PycharmProject\\LeNet-5\\data\\MNIST\\test\\0\\mnist_test_10.png'
#
# img = cv2.imread(file_name, -1)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
a = torch.arange(1, 10)
b = a.reshape(3, 3)
print(b.size()[0])
# print(b.reshape(-1, 3*3))
print(b.view(b.size()[0], -1))