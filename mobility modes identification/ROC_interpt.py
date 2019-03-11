import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interp
from sklearn import metrics

fpr = []; tpr = [];
num_fold = 10
for i in range(1,num_fold+1):
    file = pd.read_csv('CNN_model/fold'+str(i)+'/ROC_curve_fold'+str(i)+'.csv',sep=',')
    fpr.append(file['fpr'].values);
    tpr.append(file['tpr'].values);

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_fold)]))
sum_tpr = np.zeros_like(all_fpr)
for i in range(num_fold):
    sum_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr = sum_tpr / num_fold
auc_roc = metrics.auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc_roc)
plt.show()

ROC_curve = pd.DataFrame({'fpr': all_fpr, 'tpr': mean_tpr, 'auc': auc_roc})
ROC_curve.to_csv('CNN_model/ROC_curve.csv', index=False)

print('hhh')