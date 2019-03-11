import pandas as pd
import numpy as np

traj = pd.read_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_traj_with_label_grid.csv')
# traj.rename(columns={'FID_Zone18':'FID_traj', 'BASEDATETI':'BASEDATETIME', 'VESSELLABE':'VESSELLABEL', 'COMBINEDLA':'COMBINEDLABEL'},inplace=True)
# traj.sort_values('FID_traj',inplace=True)
# traj.drop_duplicates('FID_traj','first',inplace=True)
# traj = traj.reset_index(drop=True)

traj_with_label = traj[~(traj['COMBINEDLABEL']==0)]
traj_with_label = traj_with_label.reset_index(drop=True)

images = np.zeros([2358,7620])
SOG_sum = np.zeros([2358,7620])
labels = np.zeros([2358,85])

m = 0
images[m,traj_with_label['Id'][0]-1] += 1
labels[m,traj_with_label['COMBINEDLABEL'][0]-1] = 1
for i in range(1,len(traj_with_label)):
    if traj_with_label['VOYAGEID'][i] != traj_with_label['VOYAGEID'][i-1]:
        m += 1
    images[m,traj_with_label['Id'][i]-1] += 1
    SOG_sum[m,traj_with_label['Id'][i]-1] += traj_with_label['SOG'][i]
    labels[m,traj_with_label['COMBINEDLABEL'][i]-1] = 1

SOG_ave = SOG_sum/images

Images = pd.DataFrame(images)
SOG_images = pd.DataFrame(SOG_ave)
Labels = pd.DataFrame(labels)

SOG_images.fillna(0,inplace=True)

reindex = np.random.permutation(Images.index)

Images = Images.reindex(reindex)
SOG_images = SOG_images.reindex(reindex)
Labels = Labels.reindex(reindex)
Images = Images.reset_index(drop=True)
SOG_images = SOG_images.reset_index(drop=True)
Labels = Labels.reset_index(drop=True)

Images.to_csv('data/Zone18_2014_05-08/labeled_images_of_traj_combined.csv',index=False)
SOG_images.to_csv('data/Zone18_2014_05-08/labeled_SOG_images_of_traj_combined.csv',index=False)
Labels.to_csv('data/Zone18_2014_05-08/labels_of_images_of_traj_combined.csv',index=False)

print('hhh')
