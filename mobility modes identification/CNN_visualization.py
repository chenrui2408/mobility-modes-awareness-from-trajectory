import tensorflow as tf
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

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
############################feature visualization######################################
def feature_visualization(train_data,test_data,num_of_train_sample,num_of_class,fold):

    def conv2d_transpose(x, W):
        return tf.nn.conv2d_transpose(x, W, output_shape=[1,127,60,1], strides=[1,1,1,1],padding='SAME')

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    model = tf.train.import_meta_graph('CNN_model/fold'+fold+'/model_fold'+fold+'.meta')
    model.restore(sess,tf.train.latest_checkpoint('CNN_model/fold'+fold+'/'))

    y_test = np.argmax(test_data[1], 1)
    traj_num = 45 #45 or 93
    traj_label = y_test[traj_num]#46 or 47
    plt.imshow(np.reshape(test_data[0][traj_num], (127, 60))[::-1],cmap=plt.cm.gray_r)
    plt.axis('off')
    plt.savefig('D:\Cherry2.0\Desperado\Zero\Push Push 2.0\图片\卷积层可视化\label'
                +str(traj_label)+'_testdata_fold1_'+str(traj_num)+'/traj.png',dpi=600)
    plt.show()

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    # y_ = graph.get_tensor_by_name('y_:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')#keep_prob_1:0
    #
    # train_feed_dict = {x: train_data[0], y_: train_data[1], keep_prob: 1}
    # test_feed_dict = {x: test_data[0], y_: test_data[1], keep_prob: 1}
    feed_dict = {x: test_data[0][traj_num:traj_num + 1],keep_prob:1}

    # test_predict = sess.run(predict,feed_dict=test_feed_dict)[0]

    # y_pred = np.argmax(test_predict,1)

    W1 = graph.get_tensor_by_name('conv1/W:0')
    b1 = graph.get_tensor_by_name('conv1/b:0')

    W2 = graph.get_tensor_by_name('conv2/W:0')
    b2 = graph.get_tensor_by_name('conv2/b:0')

    W3 = graph.get_tensor_by_name('conv3/W:0')
    b3 = graph.get_tensor_by_name('conv3/b:0')

    conv1 = graph.get_tensor_by_name('conv1/Conv2D:0')
    pool1 = graph.get_tensor_by_name('conv1/MaxPool:0')

    # deconv1 = tf.nn.relu(conv2d_transpose(conv1,W1)+b1)
    # deconv1 = sess.run(deconv1, feed_dict=feed_dict)
    # deconv1_trans = sess.run(tf.transpose(deconv1,perm=[3,0,1,2]))

    # conv1 = sess.run(conv1, feed_dict=feed_dict)
    # conv1_trans = sess.run(tf.transpose(conv1, perm=[3, 0, 1, 2]))
    # for i in range(6):
    #     plt.imshow(conv1_trans[i][0],cmap=plt.cm.gray_r)
    #     plt.axis('off')
    #     plt.savefig('D:\Cherry2.0\Desperado\Zero\Push Push 2.0\图片\卷积层可视化\label'
    #             +str(traj_label)+'_testdata_fold1_'+str(traj_num)+'/conv1/conv1' + str(i) + '.png', dpi=300)
    #     plt.show()


    conv2 = graph.get_tensor_by_name('conv2/Conv2D:0')
    pool2 = graph.get_tensor_by_name('conv2/MaxPool:0')
    # conv2 = sess.run(conv2, feed_dict=feed_dict)
    # conv2_trans = sess.run(tf.transpose(conv2, perm=[3, 0, 1, 2]))
    # for i in range(6):
    #     plt.imshow(conv2_trans[i][0],cmap=plt.cm.gray_r)
    #     plt.axis('off')
    #     plt.savefig('D:\Cherry2.0\Desperado\Zero\Push Push 2.0\图片\卷积层可视化\label'
    #             +str(traj_label)+'_testdata_fold1_'+str(traj_num)+'/conv2/conv2' + str(i) + '.png', dpi=300)
    #     plt.show()

    conv3 = graph.get_tensor_by_name('conv3/Conv2D:0')
    pool3 = graph.get_tensor_by_name('conv3/MaxPool:0')
    # conv3 = sess.run(conv3, feed_dict=feed_dict)
    # conv3_trans = sess.run(tf.transpose(conv3, perm=[3, 0, 1, 2]))
    # for i in range(6):
    #     plt.imshow(conv3_trans[i][0],cmap=plt.cm.gray_r)
    #     plt.axis('off')
    #     plt.savefig('D:\Cherry2.0\Desperado\Zero\Push Push 2.0\图片\卷积层可视化\label'
    #             +str(traj_label)+'_testdata_fold1_'+str(traj_num)+'/conv3/conv3' + str(i) + '.png', dpi=300)
    #     plt.show()

    fc1 = graph.get_tensor_by_name('fc1/MatMul:0')
    predict = graph.get_collection('prediction')[0]
    for index,i in enumerate([pool1,pool2,pool3,fc1]):
        grad = tf.gradients(i, x)
        grad = sess.run(grad, feed_dict=feed_dict)
        grad_trans = tf.reshape(grad[0], [-1, 127, 60, 1])
        grad_trans = sess.run(tf.transpose(grad_trans, perm=[3, 0, 1, 2]))
        cmap = plt.get_cmap('Blues')
        # norm = matplotlib.colors.Normalize(vmin=-50, vmax=70)
        img = plt.imshow(grad_trans[0][0][::-1], cmap=cmap)#norm=norm
        plt.axis('off')
        cbar = plt.colorbar(img)
        # cbar.set_ticks(np.linspace(-50,70,7))
        # cbar.set_ticklabels(('-50','-30','-10','10','30','50','70'))
        plt.savefig('D:\Cherry2.0\Desperado\Zero\Push Push 2.0\图片\卷积层可视化\label'
                    + str(traj_label) + '_testdata_fold1_' + str(traj_num) + '/'+str(index)+'.png', dpi=600)
        plt.show()

    sess.close()

    print('完成卷积层可视化')

###################################################################################

if __name__ == '__main__':
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
    feature_visualization(train_data,test_data,num_of_train_sample,num_of_class,fold)