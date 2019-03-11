import pandas as pd
import math

traj = pd.read_csv('data/marinetraj_complete_500_02_grid.csv',sep=',')

lng = []; lat = []; time = []; cog = []; heading = []; mmsi = []; rot = []; sog = []; status = []; voyageid = []; id = []
grid_start = 0

for i in range(grid_start,len(traj)-1):
    if traj['VOYAGEID'][i+1] == traj['VOYAGEID'][i]:
        if (traj['Id'][i+1] != traj['Id'][i]) or (i+1 == len(traj)-1):
            if i == grid_start:
                lng.append(traj['XCoord'][i]); lat.append(traj['YCoord'][i]); time.append(traj['BASEDATETIME'][i]); cog.append(traj['COG'][i]);
                heading.append(traj['HEADING'][i]); mmsi.append(traj['MMSI'][i]); rot.append(traj['ROT'][i]); sog.append(traj['SOG'][i]);
                status.append(traj['STATUS'][i]); voyageid.append(traj['VOYAGEID'][i]); id.append(traj['Id'][i])
            else:
                media = math.ceil((i+grid_start)/2)

                lng.append(traj['XCoord'][media]); lat.append(traj['YCoord'][media]); time.append(traj['BASEDATETIME'][media]); cog.append(traj['COG'][media])
                heading.append(traj['HEADING'][media]); mmsi.append(traj['MMSI'][media]); rot.append(traj['ROT'][media]); sog.append(traj['SOG'][media])
                status.append(traj['STATUS'][media]); voyageid.append(traj['VOYAGEID'][media]); id.append(traj['Id'][media])

            grid_start = i+1
    else:
        if i == grid_start:
            lng.append(traj['XCoord'][i]); lat.append(traj['YCoord'][i]); time.append(traj['BASEDATETIME'][i]); cog.append(traj['COG'][i]);
            heading.append(traj['HEADING'][i]); mmsi.append(traj['MMSI'][i]); rot.append(traj['ROT'][i]); sog.append(traj['SOG'][i]);
            status.append(traj['STATUS'][i]); voyageid.append(traj['VOYAGEID'][i]); id.append(traj['Id'][i])
        else:
            media = math.ceil((i + grid_start) / 2)

            lng.append(traj['XCoord'][media]); lat.append(traj['YCoord'][media]); time.append(traj['BASEDATETIME'][media]); cog.append(traj['COG'][media]);
            heading.append(traj['HEADING'][media]); mmsi.append(traj['MMSI'][media]); rot.append(traj['ROT'][media]); sog.append(traj['SOG'][media]);
            status.append(traj['STATUS'][media]); voyageid.append(traj['VOYAGEID'][media]); id.append(traj['Id'][media])

        grid_start = i+1

traj_onlygrid = pd.DataFrame({'XCoord':lng,'YCoord':lat,'BASEDATETIME':time,'COG':cog,'HEADING':heading,
                              'MMSI':mmsi,'ROT':rot,'SOG':sog,'STATUS':status,'VOYAGEID':voyageid,'Id':id})
traj_onlygrid.to_csv('data/marinetraj_complete_500_02_onlygrid.csv',index=False)

print('hhh')
