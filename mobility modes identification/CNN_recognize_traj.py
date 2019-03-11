import tensorflow as tf
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import metrics
from scipy import interp

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

def CNN_train(train_data,test_data,num_of_train_sample,num_of_class,fold):
###########################变量设置：权重和偏置###########################
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial,name='W')

    def bias_variable(shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial,name='b')
###########################卷积核和池化设置：卷积核大小，移动步长#############
    def conv2d(x,W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1],
                              strides=[1,2,2,1], padding='SAME')
    def max_pool_4x4(x):
        return tf.nn.max_pool(x, ksize=[1,4,4,1],
                              strides=[1,4,4,1], padding='SAME')
###########################输入##########################################
    x = tf.placeholder('float', shape=[None, 7620],name='x')
    y_ = tf.placeholder('float', shape=[None, num_of_class],name='y_')

    x_image = tf.reshape(x, [-1,127,60,1],name='x_image')
###########################利用预训练的CAE对输入进行特征提取#################
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    #
    # model = tf.train.import_meta_graph('DAE_model/fold'+fold+'/DAEmodel_fold'+fold+'.meta')
    # model.restore(sess,tf.train.latest_checkpoint('DAE_model/fold'+fold+'/'))
    #
    # graph = tf.get_default_graph()
    # # x = graph.get_tensor_by_name('x:0')
    # # x_image = graph.get_tensor_by_name('x_image:0')
    # # y_ = graph.get_tensor_by_name('y_:0')
    # # keep_prob = graph.get_tensor_by_name('keep_prob:0')
    # #
    # # loss_CAE = graph.get_collection('loss')
    # # train_loss_CAE= sess.run(loss_CAE,feed_dict={
    # #     x: train_data[0][0:400], y_: train_data[0][0:400], keep_prob: 1})[0]#
    # # print('train_loss_CAE %g' % train_loss_CAE)
    # # reconstruct = graph.get_collection('reconstruct')[0]
    # with tf.variable_scope('DAE'):
    #     W_conv_en = graph.get_tensor_by_name('conv_en/W:0')
    #     b_conv_en = graph.get_tensor_by_name('conv_en/b:0')
    #     W_AE = sess.run(W_conv_en)
    #     b_AE = sess.run(b_conv_en)
    #     h_conv_AE = tf.nn.relu(conv2d(x_image, W_AE) + b_AE)
    #     reconstruct = tf.layers.conv2d(inputs=h_conv_AE, filters=1, kernel_size=(1, 1), padding='same', activation=None)
        # h_pool_CAE = max_pool_4x4(h_conv_CAE)
    # sess.close()
###########################网络结构##########################################
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_4x4(h_conv1)

    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_4x4(h_conv2)

    with tf.variable_scope('conv3'):
        W_conv3 = weight_variable([5,5,64,128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.leaky_relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_4x4(h_conv3)

    h_pool3_shape = h_pool3.shape.dims
    keep_prob = tf.placeholder('float',name='keep_prob')

    with tf.variable_scope('fc1'):
        W_fc1 = weight_variable([h_pool3_shape[1].value*h_pool3_shape[2].value*h_pool3_shape[3].value,1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1,h_pool3_shape[1].value*h_pool3_shape[2].value*h_pool3_shape[3].value])
        h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc2'):
        W_fc2 = weight_variable([1024,num_of_class])
        b_fc2 = bias_variable([num_of_class])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#############################损失函数和准确率,正则化##############################
    trainable_variables = tf.trainable_variables()
    all_weights = [trainable_variables[0],trainable_variables[2],trainable_variables[4],trainable_variables[6],trainable_variables[8]]

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    reg = tf.contrib.layers.apply_regularization(
        regularizer,all_weights)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv + 1e-8))

    loss = cross_entropy + reg

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
##########################训练过程设置：优化方法和学习速率，初始化########
    lr = 0.001
    # train_step = tf.train.MomentumOptimizer(lr,0.9).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
#########################过程结果记录##########################################
    epoch_list = []
    train_loss_epoch_list = []
    test_loss_epoch_list = []
    train_accuracy_epoch_list = []
    test_accuracy_epoch_list = []
    y_pred_epoch_list = []
    y_test = np.argmax(test_data[1], 1)
########################保存模型#########################################
    saver = tf.train.Saver()
    tf.add_to_collection('prediction',y_conv)
    tf.add_to_collection('accuracy',accuracy)
########################开始训练#########################################
    max_test_accuracy = 0
    for epoch in range(1000):
        batch_size = 400
        num_of_batches = math.ceil(num_of_train_sample/batch_size)
        train_dataset = Dataset(train_data[0], train_data[1])
    #####################epoch内部每次iteration记录###########################
        iteration_list = []
        train_loss_iteration_list = []
        test_loss_iteration_list = []
        train_accuracy_iteration_list = []
        test_accuracy_iteration_list = []
        y_pred_iteration_list = []
    #########################################################################
        for i in range(num_of_batches):
            batch = train_dataset.next_batch(batch_size)#get_batch(train_data,batch_size=100)

            run_feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.1}
            train_feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1}
            test_feed_dict = {x: test_data[0], y_: test_data[1], keep_prob: 1}

            y_pred = tf.argmax(y_conv.eval(feed_dict=test_feed_dict),1).eval(test_feed_dict)
            train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
            train_loss = loss.eval(feed_dict=train_feed_dict)
            test_accuracy = accuracy.eval(feed_dict=test_feed_dict)
            test_loss = loss.eval(feed_dict=test_feed_dict)
            print('epoch %d, iteration %d, training accuracy %g, training loss %g, test_accuracy %g, test loss %g'
                  %(epoch, i, train_accuracy, train_loss, test_accuracy, test_loss))#

            iteration_list.append(i);train_loss_iteration_list.append(train_loss)
            test_loss_iteration_list.append(test_loss);train_accuracy_iteration_list.append(train_accuracy)
            test_accuracy_iteration_list.append(test_accuracy);y_pred_iteration_list.append(y_pred)  #
        #########################保存CNN模型，重要！！！#####################################################
            if test_accuracy >= max_test_accuracy:
                max_test_accuracy = test_accuracy
                globel_step = epoch
                saver.save(sess, 'CNN_model/fold'+fold+'_leaky/model_fold'+fold)
        ################################################################################################
            train_step.run(feed_dict=run_feed_dict)
    ##################################记录每次epoch数据############################################
        test_accuracy_of_epoch = max(test_accuracy_iteration_list)
        max_index = test_accuracy_iteration_list.index(max(test_accuracy_iteration_list))
        train_loss_of_epoch = train_loss_iteration_list[max_index]; test_loss_of_epoch = test_loss_iteration_list[max_index]
        train_accuracy_of_epoch = train_accuracy_iteration_list[max_index]; y_pred_of_epoch = y_pred_iteration_list[max_index]

        epoch_list.append(epoch);train_loss_epoch_list.append(train_loss_of_epoch)
        test_loss_epoch_list.append(test_loss_of_epoch);train_accuracy_epoch_list.append(train_accuracy_of_epoch)
        test_accuracy_epoch_list.append(test_accuracy_of_epoch);y_pred_epoch_list.append(y_pred_of_epoch)#

    sess.close()
####################################保存训练过程数据################################################
    loss_result = pd.DataFrame({'epoch': epoch_list, 'train_loss': train_loss_epoch_list, 'test_loss': test_loss_epoch_list})
    accuracy_result = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': train_accuracy_epoch_list, 'test_accuracy': test_accuracy_epoch_list})
    loss_result.to_csv('CNN_model/fold' + fold + '_leaky/loss_fold' + fold + '.csv', index=False)
    accuracy_result.to_csv('CNN_model/fold' + fold + '_leaky/accuracy_fold' + fold + '.csv', index=False)

    plt.plot(epoch_list, train_loss_epoch_list, 'r');plt.plot(epoch_list, test_loss_epoch_list, 'b');plt.show()
    plt.plot(epoch_list, train_accuracy_epoch_list, 'r');plt.plot(epoch_list, test_accuracy_epoch_list, 'b');plt.show()
    print(max(test_accuracy_epoch_list))
    print('完成CNN训练')

def CNN_test(train_data,test_data,num_of_train_sample,num_of_class,fold):
    # x = tf.placeholder('float', shape=[None, 7620],name='x')
    # y_ = tf.placeholder('float', shape=[None, num_of_class],name='y_')
    # x_image = tf.reshape(x,[-1,127,60,1],name='x_image')
    # keep_prob = tf.placeholder('float')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    model = tf.train.import_meta_graph('Control_group/CNN_sample/fold'+fold+'/model_fold'+fold+'.meta')
    model.restore(sess,tf.train.latest_checkpoint('Control_group/CNN_sample/fold'+fold+'/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y_ = graph.get_tensor_by_name('y_:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')#keep_prob_1:0

    train_feed_dict = {x: train_data[0], y_: train_data[1], keep_prob: 1}
    test_feed_dict = {x: test_data[0], y_: test_data[1], keep_prob: 1}

    accuracy = graph.get_collection('accuracy')
    test_accuracy = sess.run(accuracy,feed_dict=test_feed_dict)[0]#
    print('test_accuracy %g' % test_accuracy)

    predict = graph.get_collection('prediction')
    test_predict = sess.run(predict,feed_dict=test_feed_dict)[0]
    y_test = np.argmax(test_data[1],1)
    y_pred = np.argmax(test_predict,1)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    F1_score = metrics.f1_score(y_test, y_pred, average='weighted')

    fpr = []; tpr = []; auc = [];
    for i in range(test_data[1].shape[1]):
        f,t,_ = metrics.roc_curve(test_data[1][:,i],test_predict[:,i])
        t = np.nan_to_num(t)
        a = metrics.auc(f,t)
        fpr.append(f);tpr.append(t);auc.append(a)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(test_data[1].shape[1])]))
    sum_tpr = np.zeros_like(all_fpr)
    for i in range(test_data[1].shape[1]):
        num = str(test_data[1][:,i].tolist()).count('1')
        sum_tpr += interp(all_fpr, fpr[i], tpr[i])*num
    mean_tpr = sum_tpr/test_data[1].shape[0]
    auc_roc = metrics.auc(all_fpr,mean_tpr)
    plt.plot(all_fpr, mean_tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc_roc);plt.show()

    result = [test_accuracy,precision,recall,F1_score,auc_roc]
    result = pd.Series(result,index=['accuracy','precision','recall','F1_score','AUC'])
    ROC_curve = pd.DataFrame({'fpr':all_fpr,'tpr':mean_tpr})

    ROC_curve.to_csv('Control_group/CNN_sample/fold'+fold+'/ROC_curve_fold' + fold + '.csv',index=False)
    result.to_csv('Control_group/CNN_sample/fold'+fold+'/result_fold' + fold + '.csv')

    sess.close()

    print('完成CNN预测')
###################################################################################


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
        traj_test = traj.iloc[i:i+fold,:]
        SOG_test = SOG.iloc[i:i+fold,:]
        label_test = label.iloc[i:i+fold,:]
        # traj_test = pd.read_csv('data/test_data_image.csv',sep=',')
        traj_train = traj.drop(traj.index[i:i+fold])
        SOG_train = SOG.drop(traj.index[i:i+fold])
        label_train = label.drop(traj.index[i:i+fold])

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


    traj = normalize_traj_data(traj)
    SOG = normalize_SOG_data(SOG)

    ten_fold_cross_validation = 1
    fold = str(ten_fold_cross_validation)
    train_data, test_data = divide_train_test_data(traj,SOG,label,ten_fold_cross_validation)

    num_of_train_sample = train_data[0].shape[0]
    num_of_class = label.shape[1]
    CNN_train(train_data,test_data,num_of_train_sample,num_of_class,fold)
    # CNN_test(train_data,test_data,num_of_train_sample,num_of_class,fold)
