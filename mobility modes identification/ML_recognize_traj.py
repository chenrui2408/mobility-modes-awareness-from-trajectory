import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interp
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib

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

###############################SVM_train#########################################
def SVM_train(train_data_pca,train_label,test_data_pca,test_label,fold):

    y_train = np.argmax(train_label, 1)
    y_test = np.argmax(test_label, 1)

    C = []; G = []; A = []
    max_acc = 0
    for c in [1000]:#1,2,5,10,20,50,100,200,500,800,1000,2000,3000,5000,10000
        for gamma in [0.08]:#0.001,0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1
            clf = SVC(C=c,decision_function_shape='ovo',gamma=gamma,probability=True)
            clf.fit(train_data_pca,y_train)
            y_pred = clf.predict(test_data_pca)

            accuracy = metrics.accuracy_score(y_test,y_pred)

            C.append(c); G.append(gamma); A.append(accuracy)
            print('C %g, gamma %g, accuracy %g'%(c,gamma,accuracy))

            if accuracy >= max_acc:
                max_acc = accuracy
                joblib.dump(clf, 'Control_group/SVM_speed/model_fold' + fold + '.pkl')

    print('optimal paramaters: C %g, gamma %g, accuracy %g'%
          (C[A.index(max(A))], G[A.index(max(A))], max(A)))

    print('完成SVM of fold'+fold+'训练')
###############################SVM_test########################################################
def SVM_test(train_data_pca,train_label,test_data_pca,test_label,fold):

    y_train = np.argmax(train_label, 1)
    y_test = np.argmax(test_label, 1)

    clf = joblib.load('Control_group/SVM_speed/model_fold'+fold+'.pkl')

    C = clf.C
    gamma = clf._gamma

    y_pred = clf.predict(test_data_pca)
    y_pred_prob = clf.predict_proba(test_data_pca)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    F1_score = metrics.f1_score(y_test, y_pred, average='weighted')

    fpr = []; tpr = []; auc = [];
    for i in range(test_label.shape[1]):
        f, t, _ = metrics.roc_curve(test_label[:, i], y_pred_prob[:, i])
        t = np.nan_to_num(t)
        a = metrics.auc(f, t)
        fpr.append(f); tpr.append(t); auc.append(a)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(test_label.shape[1])]))
    sum_tpr = np.zeros_like(all_fpr)
    for i in range(test_label.shape[1]):
        num = str(test_label[:, i].tolist()).count('1')
        sum_tpr += interp(all_fpr, fpr[i], tpr[i]) * num
    mean_tpr = sum_tpr / test_data[1].shape[0]
    auc_roc = metrics.auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc_roc);plt.show()

    print('parameter_C %g,parameter_gamma %g, accuracy %g, precision %g, recall %g, '
          'F1_score %g, AUC %g' % (C, gamma, accuracy, precision, recall, F1_score, auc_roc))
    result = [C, gamma, accuracy, precision, recall, F1_score, auc_roc]
    result = pd.Series(result, index=['parameter_C','parameter_gamma','accuracy',
                                      'precision','recall', 'F1_score','AUC'])
    ROC_curve = pd.DataFrame({'fpr': all_fpr,'tpr': mean_tpr})

    ROC_curve.to_csv('Control_group/SVM_speed/ROC_curve_fold'+fold+'.csv',index=False)
    result.to_csv('Control_group/SVM_speed/result_fold'+fold+'.csv')
    print('完成SVM of fold' + fold + '测试')
###############################DT_train########################################################
def DT_train(train_data_pca,train_label,test_data_pca,test_label,fold):

    y_train = np.argmax(train_label, 1)
    y_test = np.argmax(test_label, 1)

    D = []; A = []
    max_acc = 0
    for depth in range(15,26):#range(15,26)
        clf = DecisionTreeClassifier(class_weight='balanced',max_depth=depth,max_features=None)
        clf.fit(train_data_pca,y_train)
        y_pred = clf.predict(test_data_pca)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        D.append(depth); A.append(accuracy)
        print('depth %g, accuracy %g'%(depth,accuracy))

        if accuracy >= max_acc:
            max_acc = accuracy
            joblib.dump(clf, 'Control_group/DT_speed/model_fold'+fold+'.pkl')

    print('optimal paramaters: depth %g, accuracy %g' % (D[A.index(max(A))],max(A)))
    print('完成DT of fold'+fold+'训练')
###########################DT_test#######################################################
def DT_test(train_data_pca,train_label,test_data_pca,test_label,fold):

    y_train = np.argmax(train_label, 1)
    y_test = np.argmax(test_label, 1)

    clf = joblib.load('Control_group/DT_speed/model_fold'+fold+'.pkl')
    tree = clf.tree_
    depth = tree.max_depth
    y_pred = clf.predict(test_data_pca)
    y_pred_prob = clf.predict_proba(test_data_pca)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    F1_score = metrics.f1_score(y_test, y_pred, average='weighted')

    fpr = []; tpr = []; auc = [];
    for i in range(test_label.shape[1]):
        f, t, _ = metrics.roc_curve(test_label[:, i], y_pred_prob[:, i])
        t = np.nan_to_num(t)
        a = metrics.auc(f, t)
        fpr.append(f); tpr.append(t); auc.append(a)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(test_label.shape[1])]))
    sum_tpr = np.zeros_like(all_fpr)
    for i in range(test_label.shape[1]):
        num = str(test_label[:, i].tolist()).count('1')
        sum_tpr += interp(all_fpr, fpr[i], tpr[i]) * num
    mean_tpr = sum_tpr / test_data[1].shape[0]
    auc_roc = metrics.auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc_roc);plt.show()

    print('parameter_depth %g, accuracy %g, precision %g, recall %g, '
          'F1_score %g, AUC %g' % (depth, accuracy, precision, recall, F1_score, auc_roc))

    result = [depth, accuracy, precision, recall, F1_score, auc_roc]
    result = pd.Series(result, index=['parameter_depth','accuracy', 'precision',
                                      'recall', 'F1_score','AUC'])
    ROC_curve = pd.DataFrame({'fpr': all_fpr,'tpr': mean_tpr})

    ROC_curve.to_csv('Control_group/DT_speed/ROC_curve_fold'+fold+'.csv',index=False)
    result.to_csv('Control_group/DT_speed/result_fold'+fold+'.csv')
    print('完成DT of fold'+fold+'测试')
###############################RF_train########################################################
def RF_train(train_data_pca,train_label,test_data_pca,test_label,fold):

    y_train = np.argmax(train_label, 1)
    y_test = np.argmax(test_label, 1)

    D = []; N = []; A = []; O = []
    max_acc = 0
    for n in [60]:#range(10,110,10)
        for depth in [22]:#range(15,26)
            clf = RandomForestClassifier(n_estimators=n,max_depth=depth,max_features=None,oob_score=True,class_weight='balanced',n_jobs=4)
            clf.fit(train_data_pca,y_train)
            y_pred = clf.predict(test_data_pca)

            oob_score = clf.oob_score_
            accuracy = metrics.accuracy_score(y_test, y_pred)
            N.append(n); D.append(depth); A.append(accuracy); O.append(oob_score)
            print('n_estimators %g, depth %g, accuracy %g, oob_score %g'%(n,depth,accuracy,oob_score))

            if accuracy >= max_acc:
                max_acc = accuracy
                joblib.dump(clf, 'Control_group/RF_speed/model_fold'+fold+'.pkl')

    print('optimal paramaters: n_estimators %g, depth %g, accuracy %g, oob_score %g'
          % (N[A.index(max(A))], D[A.index(max(A))],max(A), O[A.index(max(A))]))
    print('完成RF of fold'+fold+'训练')
###########################RF_test#######################################################
def RF_test(train_data_pca,train_label,test_data_pca,test_label,fold):

    y_train = np.argmax(train_label, 1)
    y_test = np.argmax(test_label, 1)

    clf = joblib.load('Control_group/RF_speed/model_fold'+fold+'.pkl')
    n = clf.n_estimators
    depth = clf.max_depth
    y_pred = clf.predict(test_data_pca)
    y_pred_prob = clf.predict_proba(test_data_pca)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    F1_score = metrics.f1_score(y_test, y_pred, average='weighted')

    fpr = []; tpr = []; auc = [];
    for i in range(test_label.shape[1]):
        f, t, _ = metrics.roc_curve(test_label[:, i], y_pred_prob[:, i])
        t = np.nan_to_num(t)
        a = metrics.auc(f, t)
        fpr.append(f); tpr.append(t); auc.append(a)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(test_label.shape[1])]))
    sum_tpr = np.zeros_like(all_fpr)
    for i in range(test_label.shape[1]):
        num = str(test_label[:, i].tolist()).count('1')
        sum_tpr += interp(all_fpr, fpr[i], tpr[i]) * num
    mean_tpr = sum_tpr / test_data[1].shape[0]
    auc_roc = metrics.auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc_roc);plt.show()

    print('parameter_n_estimators %g, parameter_depth %g, accuracy %g, precision %g, recall %g, '
          'F1_score %g, AUC %g' % (n, depth, accuracy, precision, recall, F1_score, auc_roc))

    result = [n, depth, accuracy, precision, recall, F1_score, auc_roc]
    result = pd.Series(result, index=['parameter_n_estimators','parameter_depth','accuracy',
                                      'precision','recall', 'F1_score','AUC'])
    ROC_curve = pd.DataFrame({'fpr': all_fpr,'tpr': mean_tpr})

    ROC_curve.to_csv('Control_group/RF_speed/ROC_curve_fold'+fold+'.csv',index=False)
    result.to_csv('Control_group/RF_speed/result_fold'+fold+'.csv')
    print('完成RF of fold'+fold+'测试')
#########################################################################################


def ML(train_data,test_data,fold,task,method):

    pca_train = PCA(n_components=235)
    pca_train.fit(train_data[0])
    energy_train = pca_train.explained_variance_ratio_
    total_energy_train = np.sum(energy_train)
    train_data_pca = pca_train.transform(train_data[0])
    train_label = train_data[1]

    test_data_pca = pca_train.transform(test_data[0])
    test_label = test_data[1]

    if task == 'train':
        if method == 'SVM':
            SVM_train(train_data_pca,train_label,test_data_pca,test_label,fold)
        elif method == 'DT':
            DT_train(train_data_pca,train_label,test_data_pca,test_label,fold)
        elif method == 'RF':
            RF_train(train_data_pca,train_label,test_data_pca,test_label,fold)
        else:
            print('I beg you Pardon?')
    elif task == 'test':
        if method == 'SVM':
            SVM_test(train_data_pca,train_label,test_data_pca,test_label,fold)
        elif method == 'DT':
            DT_test(train_data_pca,train_label,test_data_pca,test_label,fold)
        elif method == 'RF':
            RF_test(train_data_pca,train_label,test_data_pca,test_label,fold)
        else:
            print('I beg you Pardon?')
    else:
        print('what do you want?')
########################################################################################

if __name__ == '__main__':
    traj = pd.read_csv('data/Zone18_2014_05-08/labeled_images_of_traj_combined.csv', sep=',')
    SOG = pd.read_csv('data/Zone18_2014_05-08/labeled_SOG_images_of_traj_combined.csv', sep=',')
    label = pd.read_csv('data/Zone18_2014_05-08/labels_of_images_of_traj_combined.csv', sep=',')

    traj = normalize_traj_data(traj)
    SOG = normalize_SOG_data(SOG)

    ten_fold_cross_validation = 1   #from 1 to 10
    fold = str(ten_fold_cross_validation)
    train_data, test_data = divide_train_test_data(traj,SOG,label,ten_fold_cross_validation)

    num_of_train_sample = train_data[0].shape[0]
    num_of_class = label.shape[1]

    task = 'test'  #train or test
    method = 'RF'   #SVM or DT or RF
    ML(train_data,test_data,fold,task,method)
