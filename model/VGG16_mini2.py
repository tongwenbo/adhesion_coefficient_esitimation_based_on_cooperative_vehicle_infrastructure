import tensorflow as tf
'''
vgg16_mini模型
'''


class vgg16:

    def __init__(self, imgs):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8

    def saver(self):
        return tf.train.Saver()

    def maxpool(self, name, input_data):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name,input_data,out_channel):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights",[3,3,in_channel,out_channel],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32))
            biases = tf.get_variable("biases",[out_channel],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32))
            conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding="SAME")
            res = tf.nn.bias_add(conv_res,biases)
            out = tf.nn.relu(res,name=name)
        return out

    def fc(self,name,input_data,out_channel):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1]*shape[-2]*shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weight",shape=[size,out_channel],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32))
            biases = tf.get_variable(name='biases',shape=[out_channel],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32))
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        return out

    def convlayers(self):
        self.conv1_1 = self.conv("conv1re_1",self.imgs,8)
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,8)
        self.pool1 = self.maxpool("poolre1",self.conv1_2)

        self.conv2_1 = self.conv("conv2_1",self.pool1,16)
        self.conv2_2 = self.conv("convwe2_2",self.conv2_1,16)
        self.pool2 = self.maxpool("pool2",self.conv2_2)


    #全连接层
    def fc_layers(self):
        self.fc6 = self.fc("fc1",self.pool2,128)
        self.fc8 = self.fc("fc3",self.fc6,7)#n_class是输出类别数，比如猫狗分类这里改为2即可，10分类改为10

'''


'''