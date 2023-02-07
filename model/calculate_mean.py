import tensorflow as tf
import os

'''
计算训练图片均值
'''


def calc_mean_variance(file_dir, pic_width, pic_height):
    """
        计算均值和方差
    :return:
    """
    images = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
    print("第一张图片为：", images[0])
    img_all = None
    for image_path in images:
        image = tf.cast(image_path, tf.string)  # tf.cast类型转换
        image_contents = tf.read_file(image)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image = tf.expand_dims(image, 0)
        if img_all is None:
            img_all = image
        else:
            img_all = tf.concat([img_all, image], 0)

    img_all = tf.cast(img_all, tf.float32)
    axes = list(range(len(img_all.get_shape()) - 1))
    mean, variance = tf.nn.moments(img_all, axes=axes)  # axes=[0,1,2]即在0,1,2上进行均值方差计算
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("均值：", sess.run(mean))
        print("方差：", sess.run(variance))


if __name__ == "__main__":
    # calc_mean_variance("./data/train",448,32)  # 均值：[42.79902 42.79902 42.79902] 方差：[10956.549 10935.391 10956.549]
    calc_mean_variance("./data/val", 448, 32)  # 均值： [175.05261 162.04726 158.91115]方差： [3135.91   3010.6238 2997.9714]
