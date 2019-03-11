import pandas as pd
import math

def complete(traj,interval_threshold):
    lng = []; lat = []; time = []; cog = []; heading = []; mmsi = []; rot = []; sog = []; status = []; voyageid = []
    lng.append(traj['XCoord'][0]); lat.append(traj['YCoord'][0]); time.append(traj['BASEDATETIME'][0]); cog.append(traj['COG'][0])
    heading.append(traj['HEADING'][0]); mmsi.append(traj['MMSI'][0]); rot.append(traj['ROT'][0]); sog.append(traj['SOG'][0])
    status.append(traj['STATUS'][0]); voyageid.append(traj['VOYAGEID'][0])
    for i in range(0,len(traj)-1):
        if traj['VOYAGEID'][i+1] != traj['VOYAGEID'][i]:
            lng.append(traj['XCoord'][i+1]); lat.append(traj['YCoord'][i+1]); time.append(traj['BASEDATETIME'][i+1]); cog.append(traj['COG'][i+1])
            heading.append(traj['HEADING'][i+1]); mmsi.append(traj['MMSI'][i+1]); rot.append(traj['ROT'][i+1]); sog.append(traj['SOG'][i+1])
            status.append(traj['STATUS'][i+1]); voyageid.append(traj['VOYAGEID'][i+1])
            continue
        else:
            distance = math.sqrt(math.pow(traj['XCoord'][i+1]-traj['XCoord'][i],2)+math.pow(traj['YCoord'][i+1]-traj['YCoord'][i],2))
            if distance <= interval_threshold:
                lng.append(traj['XCoord'][i+1]); lat.append(traj['YCoord'][i+1]); time.append(traj['BASEDATETIME'][i+1]); cog.append(traj['COG'][i+1])
                heading.append(traj['HEADING'][i+1]); mmsi.append(traj['MMSI'][i+1]); rot.append(traj['ROT'][i+1]); sog.append(traj['SOG'][i+1])
                status.append(traj['STATUS'][i+1]); voyageid.append(traj['VOYAGEID'][i+1])
                continue
            else:
                for j in range(0,int(distance/interval_threshold)):
                    lng.append(float('nan')); lat.append(float('nan')); time.append(float('nan')); cog.append(float('nan'))
                    heading.append(float('nan')); mmsi.append(float('nan')); rot.append(float('nan')); sog.append(float('nan'))
                    status.append(traj['STATUS'][i]); voyageid.append(traj['VOYAGEID'][i])

                lng.append(traj['XCoord'][i+1]); lat.append(traj['YCoord'][i+1]); time.append(traj['BASEDATETIME'][i+1]); cog.append(traj['COG'][i+1])
                heading.append(traj['HEADING'][i+1]); mmsi.append(traj['MMSI'][i+1]); rot.append(traj['ROT'][i+1]); sog.append(traj['SOG'][i+1])
                status.append(traj['STATUS'][i+1]); voyageid.append(traj['VOYAGEID'][i+1])

    traj_complete = pd.DataFrame({'XCoord':lng,'YCoord':lat,'BASEDATETIME':time,'COG':cog,'HEADING':heading,
                              'MMSI':mmsi,'ROT':rot,'SOG':sog,'STATUS':status,'VOYAGEID':voyageid})
    traj_complete = traj_complete.interpolate()

    return traj_complete

    print('hhh')

def main():
    traj = pd.read_csv('data/Zone18_2014_05/Zone18_2014_05_cleaned_500.csv',sep=',')
    interval_threshold = 0.02
    traj_complete = complete(traj,interval_threshold)
    traj_complete.to_csv('data/Zone18_2014_05/Zone18_2014_05_complete_500_02.csv',index=False)

    print('hhh')

if __name__ == '__main__':
    main()