import pandas as pd

traj = pd.read_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_complete_500_02.csv',sep=',')
voyage = pd.read_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_Origin_Destination.csv',sep=',')
# vessel = pd.read_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_VesselType.csv',)

# MMSI = traj[['VOYAGEID','MMSI']];
# # MMSI.drop_duplicates(['VOYAGEID'],inplace=True)
# # MMSI.reset_index(drop=True,inplace=True)

label = voyage['Label'].values.tolist()
vessellabel = voyage['VesselLabel'].values.tolist()
combinedlabel = voyage['CombinedLabel'].values.tolist()

# voyage = voyage.join(MMSI.set_index('VOYAGEID'),on='VOYAGEID')
# voyage = voyage.join(vessel.set_index('MMSI'),on = 'MMSI')

# new_vessellabel = []
# for i in range(0,len(voyage-1)):
#     if voyage['VesselLabel'][i] == 2:
#         new_vessellabel.append(1)
#     elif voyage['VesselLabel'][i] == 3:
#         new_vessellabel.append(2)
#     elif voyage['VesselLabel'][i] == 4 or voyage['VesselLabel'][i] == 5:
#         new_vessellabel.append(3)
#     elif voyage['VesselLabel'][i] == 10:
#         new_vessellabel.append(4)
#     elif voyage['VesselLabel'][i] == 15:
#         new_vessellabel.append(5)
#     elif voyage['VesselLabel'][i] == 22:
#         new_vessellabel.append(6)
#     elif voyage['VesselLabel'][i] == 23:
#         new_vessellabel.append(7)
#     elif voyage['VesselLabel'][i] == 24:
#         new_vessellabel.append(8)
#     else:
#         new_vessellabel.append(0)
#
# voyage['VesselLabel'] = new_vessellabel

# voyage.to_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_Origin_Destination.csv',index=False)


def add_label(x,attributes):
    return attributes[x-1]

traj['CLUSTER'] = traj['VOYAGEID'].apply(lambda x: add_label(x,label))
traj['VESSELLABEL'] = traj['VOYAGEID'].apply(lambda x: add_label(x,vessellabel))
traj['COMBINEDLABEL'] = traj['VOYAGEID'].apply(lambda x: add_label(x,combinedlabel))

traj.to_csv('data/Zone18_2014_05-08/Zone18_2014_05-08_traj_with_label.csv',index=False)

print('hhh')