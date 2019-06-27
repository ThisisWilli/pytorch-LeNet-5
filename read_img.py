'''
@project : LeNet-5
@author  : Hoodie_Willi
@description: ${}
@time   : 2019-06-26 19:12:49
'''
import cv2
from PIL import Image
file_name = 'D:\\PycharmProject\\LeNet-5\\data\\MNIST\\test\\0\\mnist_test_10.bmp'

img = cv2.imread(file_name)
print(img.shape)
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imwrite(file_name, img)
print(gray.shape)