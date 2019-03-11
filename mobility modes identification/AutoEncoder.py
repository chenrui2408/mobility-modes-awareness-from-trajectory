import tensorflow as tf
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import math

class Dataset:
    def __init__(self,data,label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        self._num_examples = data.shape[0]
        pass
    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle index
            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch
            data_new_part =  self._data[start:end]
            label_new_part =  self._label[start:end]
            return (np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((label_rest_part, label_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return (self._data[start:end], self._label[start:end])

def autoencoder(train_data, test_data, num_of_train_sample, num_of_class,fold):

    ###########################变量设置：权重和偏置###########################
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,name='W')

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name='b')

    ###########################卷积核和池化设置：卷积核大小，移动步长#############
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv2d_transpose(x, W):
        return tf.nn.conv2d_transpose(x, W, output_shape=[400,127,60,1], strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_4x4(x):
        return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                              strides=[1, 4, 4, 1], padding='SAME')

    ###########################网络结构##########################################
    x = tf.placeholder('float', shape=[None, 7620],name='x')
    y_ = tf.placeholder('float', shape=[None, 7620],name='y_')
    mask = tf.placeholder('float', shape=[None,7620],name='mask')
    x_masked = x * mask

    x_image = tf.reshape(x_masked, [-1,127,60,1],name='x_image')
    y_image = tf.reshape(y_, [-1,127,60,1],name='y_image')
    keep_prob = tf.placeholder('float',name='keep_prob')

    with tf.variable_scope('conv_en'):
        W_en = weight_variable([5,5,1,32])
        b_en = bias_variable([32])
        conv_en = tf.nn.relu(conv2d(x_image, W_en) + b_en)
        pool_en = max_pool_2x2(conv_en)
        pool_en_drop = tf.nn.dropout(pool_en, keep_prob)


    with tf.variable_scope('conv_de'):
        pool_de = tf.image.resize_images(pool_en_drop, size=[127, 60], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        W_de = weight_variable([5,5,1,32])
        b_de = bias_variable([1])
        conv_de = tf.nn.relu(conv2d_transpose(pool_de, W_de) + b_de)
        # logits = tf.layers.conv2d(inputs=conv_de, filters=1, kernel_size=(1,1), padding='same', activation=None)

    decoded = tf.nn.sigmoid(conv_de)
    #############################损失函数和准确率,正则化##############################
    trainable_variables = tf.trainable_variables()
    all_weights = [trainable_variables[0]]

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    reg = tf.contrib.layers.apply_regularization(
        regularizer, all_weights)

    cross_entropy = -tf.reduce_sum(y_image * tf.log(decoded + 1e-8))#tf.nn.softmax_cross_entropy_with_logits(labels=y_image,logits=logits)

    loss = cross_entropy + reg
    ##########################训练过程设置：优化方法和学习速率，初始化########
    lr = 0.001
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #########################过程结果记录##########################################
    epoch_list = []
    train_loss_epoch_list = []
    ########################保存模型#########################################
    saver = tf.train.Saver()
    tf.add_to_collection('reconstruct', decoded)
    tf.add_to_collection('loss', loss)
########################开始训练#########################################
    min_loss = 10000000
    for epoch in range(300):
        batch_size = 400
        num_of_batches = math.ceil(num_of_train_sample/batch_size)
        train_dataset = Dataset(train_data[0], train_data[1])

        iteration_list = []
        train_loss_iteration_list = []

        for i in range(num_of_batches):
            batch = train_dataset.next_batch(batch_size)
            mask_np = np.random.binomial(1,0.5,batch[0].shape)
            train_loss = loss.eval(feed_dict={
                x: batch[0], y_: batch[0], mask: mask_np, keep_prob: 1})
            print('epoch %d, iteration %d, training loss %g'
                  %(epoch, i, train_loss))

            iteration_list.append(i);
            train_loss_iteration_list.append(train_loss)

            if train_loss < min_loss:
                min_loss = train_loss
                globel_step = epoch
                saver.save(sess, 'DAE_model/fold'+fold+'/DAEmodel_fold'+fold)

            train_step.run(feed_dict={x: batch[0], y_: batch[0], mask: mask_np, keep_prob: 0.1})

        train_loss_of_epoch = min(train_loss_iteration_list)
        epoch_list.append(epoch);train_loss_epoch_list.append(train_loss_of_epoch)

    loss_result = pd.DataFrame({'epoch': epoch_list, 'loss': train_loss_epoch_list})
    loss_result.to_csv('DAE_MODEL/fold'+fold+'/loss_fold'+fold+'.csv', index=False)
    plt.plot(epoch_list, train_loss_epoch_list, 'r');plt.show()
    print(min(train_loss_epoch_list))
    print('完成CAE训练')

if __name__ == '__main__':
    #######################数据准备：归一化，分割训练数据和测试数据##################
    def normalize_traj_data(traj):
        traj_value = traj.values
        row = traj_value.shape[0]
        col = traj_value.shape[1]
        traj_max_row = traj_value.max(axis=1)
        for i in range(0, row):
            for j in range(0, col):
                traj_value[i, j] = traj_value[i, j] / traj_max_row[i]
        traj = pd.DataFrame(traj_value)
        return traj


    def normalize_SOG_data(dataframe):
        data_matrix = dataframe.values
        new_data = []
        for i in range(len(data_matrix)):
            new_data.append(max(data_matrix[i]))
        data_matrix /= max(new_data)
        dataframe_norm = pd.DataFrame(data_matrix)
        return dataframe_norm


    def divide_train_test_data(traj, SOG, label,ten_fold_cross_validation):
        i = ten_fold_cross_validation-1
        fold = 235
        traj_test = traj.iloc[i:i + fold, :]
        SOG_test = SOG.iloc[i:i + fold, :]
        label_test = label.iloc[i:i + fold, :]
        # traj_test = pd.read_csv('data/test_data_image.csv',sep=',')
        traj_train = traj.drop(traj.index[i:i + fold])
        SOG_train = SOG.drop(traj.index[i:i + fold])
        label_train = label.drop(traj.index[i:i + fold])

        traj_train = traj_train.reset_index(drop=True)
        SOG_train = SOG_train.reset_index(drop=True)
        label_train = label_train.reset_index(drop=True)
        traj_test = traj_test.reset_index(drop=True)
        SOG_test = SOG_test.reset_index(drop=True)
        label_test = label_test.reset_index(drop=True)

        # train_matrix = np.zeros([traj_train.values.shape[0],traj_train.values.shape[1],2])
        # train_matrix[:,:,0] = traj_train.values;train_matrix[:,:,1] = SOG_train.values
        #
        # test_matrix = np.zeros([traj_test.values.shape[0],traj_test.values.shape[1],2])
        # test_matrix[:,:,0] = traj_test.values;test_matrix[:,:,1] = SOG_test.values

        train_data = (SOG_train.values, label_train.values)
        test_data = (SOG_test.values, label_test.values)
        return train_data, test_data

    ############################################################################
    traj = pd.read_csv('data/Zone18_2014_05-08/labeled_images_of_traj_combined.csv', sep=',')
    SOG = pd.read_csv('data/Zone18_2014_05-08/labeled_SOG_images_of_traj_combined.csv', sep=',')
    label = pd.read_csv('data/Zone18_2014_05-08/labels_of_images_of_traj_combined.csv', sep=',')
    num_of_class = label.shape[1]

    ten_fold_cross_validation = 1
    fold = str(ten_fold_cross_validation)

    traj = normalize_traj_data(traj)
    SOG = normalize_SOG_data(SOG)
    train_data, test_data = divide_train_test_data(traj, SOG, label,ten_fold_cross_validation)

    num_of_train_sample = train_data[0].shape[0]
    autoencoder(train_data, test_data, num_of_train_sample, num_of_class,fold)
