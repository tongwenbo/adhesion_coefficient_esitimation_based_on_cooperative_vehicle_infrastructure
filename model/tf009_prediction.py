import tensorflow as tf
import numpy as np
import vgg16.VGG16_mini as model
import os
import cv2


scale_pic_width,scale_pic_height = 448,32
x = tf.placeholder(tf.float32,[None,scale_pic_height,scale_pic_width,3])

sess =tf.Session()
vgg = model.vgg16(x)
fc8_finetuining = vgg.probs

saver = tf.train.Saver()
print("model restoring")
saver.restore(sess,"./model/epoch000034.ckpt")

rootdir = './data/test/'
images = []

for root,sub_folders,files in os.walk(rootdir):
    for name in files:
        images.append(os.path.join(root,name))
labels = []
lh1_right,lh2_right,lh3_right,lh4_right,lh5_right,lh6_right,lh7_right = 0,0,0,0,0,0,0
lh1_wrong,lh2_wrong,lh3_wrong,lh4_wrong,lh5_wrong,lh6_wrong,lh7_wrong = 0,0,0,0,0,0,0
count = 0
for label_path in images:
    count+=1
    if count == 1 or count % 100 ==0:
        print("deal:",count,"个")
    letter = label_path.split("\\")[-2].split("/")[-1]

    image_contents = tf.read_file(label_path)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.cast(image, tf.float32)
    image -= [42.79902, 42.79902, 42.79902]  # 减均值
    image.set_shape((scale_pic_height, scale_pic_width, 3))

    image = sess.run(image)

    prob = sess.run(fc8_finetuining, feed_dict={x: [image]})
    max_index = np.argmax(prob)

    if letter =="lh1":
        if max_index == 0:lh1_right+=1
        else:lh1_wrong+=1
    elif letter =="lh2":
        if max_index == 1:lh2_right+=1
        else:lh2_wrong+=1
    elif letter == "lh3":
        if max_index == 2:lh3_right+=1
        else:lh3_wrong+=1
    elif letter == "lh4":
        if max_index == 3:lh4_right+=1
        else:lh4_wrong+=1
    elif letter == "lh5":
        if max_index == 4:lh5_right+=1
        else:lh5_wrong+=1
    elif letter == "lh6":
        if max_index == 5:lh6_right+=1
        else:lh6_wrong+=1
    elif letter == "lh7":
        if max_index == 6:lh7_right += 1
        else:lh7_wrong += 1

print("lh1类型：正确",lh1_right,"个，总数",(lh1_right+lh1_wrong),"个，正确率：",((lh1_right)/(lh1_right+lh1_wrong)))
print("lh2类型：正确",lh2_right,"个，总数",(lh2_right+lh2_wrong),"个，正确率：",((lh2_right)/(lh2_right+lh2_wrong)))
print("lh3类型：正确",lh3_right,"个，总数",(lh3_right+lh3_wrong),"个，正确率：",((lh3_right)/(lh3_right+lh3_wrong)))
print("lh4类型：正确",lh4_right,"个，总数",(lh4_right+lh4_wrong),"个，正确率：",((lh4_right)/(lh4_right+lh4_wrong)))
print("lh5类型：正确",lh5_right,"个，总数",(lh5_right+lh5_wrong),"个，正确率：",((lh5_right)/(lh5_right+lh5_wrong)))
print("lh6类型：正确",lh6_right,"个，总数",(lh6_right+lh6_wrong),"个，正确率：",((lh6_right)/(lh6_right+lh6_wrong)))
print("lh7类型：正确",lh7_right,"个，总数",(lh7_right+lh7_wrong),"个，正确率：",((lh7_right)/(lh7_right+lh7_wrong)))
print("总计：正确",(lh1_right+lh2_right+lh3_right+lh4_right+lh5_right+lh6_right+lh7_right),"个，总数",
      (lh1_right+lh2_right+lh3_right+lh4_right+lh5_right+lh6_right+lh7_right+lh1_wrong+lh2_wrong+lh3_wrong+lh4_wrong+lh5_wrong+lh6_wrong+lh7_wrong),
      "个，正确率：",((lh1_right+lh2_right+lh3_right+lh4_right+lh5_right+lh6_right+lh7_right)/(lh1_right+lh2_right+lh3_right+lh4_right+lh5_right+lh6_right+lh7_right+lh1_wrong+lh2_wrong+lh3_wrong+lh4_wrong+lh5_wrong+lh6_wrong+lh7_wrong)))

'''
'''