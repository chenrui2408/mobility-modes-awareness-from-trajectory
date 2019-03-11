import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from haversine import haversine

traj = pd.read_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_complete_500_02.csv', sep=',')
O_lat = []; O_lng = []; O_voyage = []; D_lat = []; D_lng = []; D_voyage = []
num_voyage = traj['VOYAGEID'].tail(1).values[0]

O_lng.append(traj['XCoord'][0])
O_lat.append(traj['YCoord'][0])
O_voyage.append(traj['VOYAGEID'][0])
for i in range(1,len(traj)-1):
    if traj['VOYAGEID'][i] != traj['VOYAGEID'][i-1]:
        D_lng.append(traj['XCoord'][i-1])
        D_lat.append(traj['YCoord'][i-1])
        D_voyage.append(traj['VOYAGEID'][i-1])
        O_lng.append(traj['XCoord'][i])
        O_lat.append(traj['YCoord'][i])
        O_voyage.append(traj['VOYAGEID'][i])
D_lng.append(traj['XCoord'][len(traj)-1])
D_lat.append(traj['YCoord'][len(traj)-1])
D_voyage.append(traj['VOYAGEID'][len(traj)-1])

olng = np.array(O_lng); olat = np.array(O_lat); ovoyage = np.array(O_voyage)
dlng = np.array(D_lng); dlat = np.array(D_lat); dvoyage = np.array(D_voyage)
OD_mat = np.zeros([num_voyage,8])
OD_mat[:,0] = olng; OD_mat[:,1] = olat; OD_mat[:,2] = dlng; OD_mat[:,3] = dlat; OD_mat[:,4] = dvoyage

OD = pd.DataFrame(OD_mat,columns=['Olng','Olat','Dlng','Dlat','VOYAGEID','Ocluster','Dcluster','Label'])

def distance_calc(lng, lat):
    distance_matrix = np.zeros([len(lat),len(lat)])
    for i in range(0,len(lat)):
        for j in range(i,len(lat)):
            distance_matrix[i,j] = haversine((lat[i],lng[i]),(lat[j],lng[j]))

    distance_matrix = np.triu(distance_matrix)
    distance_matrix += distance_matrix.T - np.diag(distance_matrix)

    return distance_matrix

# def cluster(distance_matrix):
#
#     print('perform DBSCAN')
#
#     eps = []; min_samples = []; num_cluster = []; s_score = []
#
#     for i in range(35,36):
#         for j in range(4,5):
#             db = DBSCAN(eps=i, min_samples=j, metric='precomputed',algorithm='auto', n_jobs=4).fit(distance_matrix)
#             label = db.labels_
#             n_clusters = len(set(label)) - (1 if -1 in label else 0)
#             if n_clusters > 1:
#                 S_score =  metrics.silhouette_score(distance_matrix, label, metric='precomputed')
#             else:
#                 S_score = -1
#
#             eps.append(i); min_samples.append(j); num_cluster.append(n_clusters); s_score.append(S_score)
#
#             print('eps= %d, min_samples= %d, number of clusters: %d, S_score: %f'
#                     % (i, j, n_clusters, S_score))
#     # paras_resluts = pd.DataFrame({'eps': eps, 'min_samples': min_samples,
#     #                             'num_clusters': num_cluster, 'S_score': s_score})
#     # paras_resluts.to_csv('data/ODcluster_para.csv')
#     return label

# OD = pd.read_csv('data/Origin_Destination.csv',sep=',')

O_distance_matrix = distance_calc(OD['Olng'], OD['Olat'])
D_distance_matrix = distance_calc(OD['Dlng'], OD['Dlat'])

OD.to_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_Origin_Destination.csv',index=False)
np.savetxt('data/Zone18_2014_05-08/Zone18_2014_05-08_Origin_distance_matrix.csv',O_distance_matrix,delimiter=',')
np.savetxt('data/Zone18_2014_05-08/Zone18_2014_05-08_Destination_distance_matrix.csv',D_distance_matrix,delimiter=',')

# O_distance_matrix = np.loadtxt(open("data/Origin_distance_matrix.csv","rb"),delimiter=",",skiprows=0)
# D_distance_matrix = np.loadtxt(open("data/Destination_distance_matrix.csv","rb"),delimiter=",",skiprows=0)

# Ocluster = cluster(O_distance_matrix)
# Dcluster = cluster(D_distance_matrix)
#
# OD['Ocluster'] = Ocluster
# OD['Dcluster'] = Dcluster


print('hhh')



