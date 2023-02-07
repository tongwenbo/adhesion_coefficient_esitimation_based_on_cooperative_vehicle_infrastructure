import os
import cv2


def save_train_pic(file_name, classfy, scale_pic_width, scale_pic_height, file_dir):
    '''
    存储训练图片至指定路径
    :param file_name: 文件名
    :param classfy: 类别名
    :param scale_pic_width: 缩放图像宽
    :param scale_pic_height: 缩放图像高
    :param file_dir: 输出图像路径
    :return:
    '''

    image = cv2.imread(file_name)
    image = cv2.resize(image, (scale_pic_width, scale_pic_height))
    # 存储图像
    cv2.imwrite(file_dir + classfy + "/" + file_name.split("\\")[-1].split(".")[0] + '.JPG', image)


def get_file(file_dir_input, file_dir_output, pic_width, pic_height):
    '''
    将图片转为224×224储存
    :param file_dir_input: 原图路径
    :param file_dir_output: 输出图像路径
    :param pic_width: 缩放图像宽
    :param pic_height:  缩放图像高
    :return:
    '''
    images = []
    for root, sub_folders, files in os.walk(file_dir_input):
        for name in files:
            images.append(os.path.join(root, name))
    print(images[0], images[0].split("\\")[-2].split("/")[-1])
    for label_name in images:
        letter = label_name.split("\\")[-2].split("/")[-1]
        save_train_pic(label_name, letter, pic_width, pic_height, file_dir_output)


# 修改照片大小再执行训练
# get_file('./data/train2/','./data/train/',448,32)  # 输入文件路径将文件转为指定大小储存储存
# get_file('./data/val1/','./data/val/',448,32)  # 验证集同步更新
get_file('./data/test1/', './data/test/', 448, 32)
