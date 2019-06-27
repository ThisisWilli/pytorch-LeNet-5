'''
@project : torch_execution
@author  : Hoodie_Willi
@description: $将训练集和测试集制作索引
@time   : 2019-06-23 19:28:38
'''
# coding:utf-8
import os

'''
    为数据集生成对应的txt文件
'''

train_txt_path = './MNIST//txt//train.txt'
train_dir = '../data//MNIST//train/'

test_txt_path = './MNIST//txt//test.txt'
test_dir = '../data//MNIST//test/'

valid_txt_path = './MNIST//txt//valid.txt'
valid_dir = '../data//MNIST//valid/'


def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    '''
    @root 当前遍历的文件夹本身的地址
    @dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    @files files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    '''
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            # 将两个路径拼接起来
            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                    continue
                label = sub_dir # 需要根据文件的名称进行切割
                # if label == 'cat':
                #     label = '1'
                # elif label == 'dog':
                #     label = '2'
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                f.write(line)
    f.close()


if __name__ == '__main__':
    gen_txt(train_txt_path, train_dir)
    gen_txt(test_txt_path, test_dir)
    gen_txt(valid_txt_path, valid_dir)