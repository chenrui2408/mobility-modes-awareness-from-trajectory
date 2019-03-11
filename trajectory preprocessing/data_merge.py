import pandas as pd
import numpy as np

traj05 = pd.read_csv('data/Zone18_2014_05/Zone18_2014_05_complete_500_02.csv', sep=',')
traj06 = pd.read_csv('data/Zone18_2014_06/Zone18_2014_06_complete_500_02.csv', sep=',')
traj07 = pd.read_csv('data/Zone18_2014_07/marinetraj_complete_500_02.csv', sep=',')
traj08 = pd.read_csv('data/Zone18_2014_08/Zone18_2014_08_complete_500_02.csv', sep=',')

traj_list = list([traj05,traj06,traj07,traj08])

traj = traj_list[0]
for i in range(1,len(traj_list)):
    length = len(traj_list[i])
    voyageid = traj_list[i-1]['VOYAGEID'].tail(1).values
    add_voyageid = [voyageid for j in range(length)]
    df_add_voyageid = pd.DataFrame(add_voyageid,columns=['VOYAGEID'])
    a = traj_list[i].add(df_add_voyageid)
    traj_list[i]['VOYAGEID'] = a['VOYAGEID']
    traj = traj.append(traj_list[i])

traj.reset_index(drop=True,inplace=True)
traj.to_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_complete_500_02.csv',index=False)
print('hhh')

