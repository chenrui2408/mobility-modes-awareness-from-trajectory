# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib.pylab import datestr2num
from pyproj import Proj, transform
from math import sqrt, pow
import numpy as np

def main():
    traj_raw = pd.read_csv('data/Zone18_2014_05/Zone18_2014_05.csv',sep=',')
    traj_raw.rename(columns = {'POINT_X': 'XCoord', 'POINT_Y': 'YCoord',
                               'Heading': 'HEADING', 'BaseDateTime': 'BASEDATETIME',
                               'Status': 'STATUS', 'VoyageID':'VOYAGEID'}, inplace=True)

    traj_raw = traj_raw.drop(columns=['OBJECTID'])
    traj_sorted = sort_MMSI(traj_raw)                                                     #筛选特定海域的，以及满足各种特定限制范围的正常数据
    traj_sorted.to_csv('data/Zone18_2014_05/Zone18_2014_05_sort.csv',index=False)
    #traj_sorted = pd.read_csv('data/Zone18_2014_05/Zone18_2014_05_sort.csv',sep=',')
    traj_transform = coord_transform(traj_sorted,False)                                   #WGS84地理坐标系转换到xy投影坐标系，方便计算直线距离
    traj_velocity = velocity_calc(traj_transform)                                         #利用船舶位置重新计算速度，对单位为m/s的船速进行二次筛选，去掉坐标突变的轨迹点
    traj_velocity.to_csv('data/Zone18_2014_05/Zone18_2014_05_noVoyageID.csv',index=False)
    # traj_velocity = pd.read_csv('data/Zone18_2014_05/Zone18_2014_05_noVoyageID.csv',sep=',')
    traj_noshortMMSI = delete_shortMMSI(traj_velocity)                                    #取长度大于500个数据点的船舶
    traj_noshortMMSI.to_csv('data/Zone18_2014_05/Zone18_2014_05_noVoyageID_noshortMMSI.csv',index=False)
    # traj_noshortMMSI = pd.read_csv('data/Zone18_2014_05/Zone18_2014_05_noVoyageID_noshortMMSI.csv', sep=',')
    traj_VoyageID = incode_VoyageID(traj_noshortMMSI)                                     #根据速度大小和时间间隔识别停留点进行轨迹段划分并编号VOYAGEID
    traj_VoyageID.to_csv('data/Zone18_2014_05/Zone18_2014_05_VoyageID.csv',index=False)
    # traj_VoyageID = pd.read_csv('data/Zone18_2014_05/Zone18_2014_05_VoyageID.csv',sep=',')
    traj_cleaned_VoyageID = sort_VoyageID(traj_VoyageID)                                    #去除停留点，根据长度筛选轨迹，VOYAGEID重新排序，递增
    traj_cleaned_VoyageID.to_csv('data/Zone18_2014_05/Zone18_2014_05_cleaned_500.csv', index=False)
    print('哈哈哈,完了！')

def sort_MMSI(traj):
    ##############筛选特定海域的，以及满足各种特定限制范围的正常数据####################
    traj = traj[(traj['YCoord']<=43.2)&(traj['YCoord']>=29.7)]      #海域范围北纬29.7-43.2之间
    traj = traj[(traj['STATUS']<=8)]                                 #正常航行或者停泊状态
    traj = traj[(traj['HEADING']<360)]                               #船头朝向范围[0,360）
    traj = traj[(traj['SOG']<=50)]                                   #速度小于50节
    traj = traj[~((traj['SOG']==0)&(traj['STATUS']==0))]            #船速为0，但是状态却为航行的错误数据
    ###################按照MMSI和时间顺序，排列出每条轨迹############################
    a = traj['BASEDATETIME']
    traj['BASEDATETIME'] = datestr2num(a)
    traj= traj.sort_values(by=['MMSI','BASEDATETIME'])
    traj = traj.reset_index(drop=True)

    return traj

def coord_transform(traj,inverse):
    ##########WGS84地理坐标系转换到xy投影坐标系，方便计算直线距离##############
    p1 = Proj(init='epsg:4326')     #GCS_WGS_1984
    p2 = Proj(init='epsg:3857')     #WGS_1984_Web_Mercator_Auxiliary_Sphere
    if inverse == False:
        x2, y2 = transform(p1, p2, traj['XCoord'].values,traj['YCoord'].values)
        traj['XCoord'] = x2
        traj['YCoord'] = y2
    else:
        x2, y2 = transform(p2, p1, traj['XCoord'].values, traj['YCoord'].values)
        traj['XCoord'] = x2
        traj['YCoord'] = y2

    return traj

def velocity_calc(traj):
    #######################利用船舶位置重新计算速度################################
    SOG = []
    for i in range(0,len(traj)-1):
        if traj['MMSI'][i] == traj['MMSI'][i+1]:
            sog = sqrt(pow(traj['XCoord'][i+1]-traj['XCoord'][i],2)+pow(traj['YCoord'][i+1]-traj['YCoord'][i],2))\
                            /((traj['BASEDATETIME'][i+1]-traj['BASEDATETIME'][i])*86400)
        else:
            sog = traj['SOG'][i]*0.514444
        SOG.append(sog)
    SOG.append(traj['SOG'][len(traj)-1]*0.514444)

    traj['SOG'] = SOG
    # Sog = sorted(SOG,reverse=True)
    ##################对单位为m/s的船速进行二次筛选，去掉坐标突变的轨迹点#######
    traj = traj[(traj['SOG']<=24)]
    traj = traj[~((traj['SOG']==0)&(traj['STATUS']==0))]
    traj = traj.reset_index(drop=True)

    return traj

def delete_shortMMSI(traj):
    ##########################取长度大于500个数据点的船舶#######################
    unique_MMSI = traj['MMSI'].value_counts()                 #统计各MMSI数据的长度
    short_MMSI = unique_MMSI[unique_MMSI<500].index.tolist()
    traj = traj[(True^traj['MMSI'].isin(short_MMSI))]        #删除数据长度小于500的MMSI
    traj = traj.reset_index(drop=True)

    return traj

def incode_VoyageID(traj):
    ###################根据速度大小识别停留点和时间间隔，进行轨迹段划分并编号VOYAGEID################
    traj['stop/move'] = np.where(traj['SOG']>0.5,0,1)

    for i in range(1,len(traj)-1):
        if (traj['stop/move'][i] != traj['stop/move'][i-1])&(traj['stop/move'][i] != traj['stop/move'][i+1]):
            traj['stop/move'][i] = traj['stop/move'][i-1]

    voyageid = []
    id = 0
    voyageid.append(0)
    for i in range(1,len(traj)):
        time_interval = traj['BASEDATETIME'][i] - traj['BASEDATETIME'][i-1]
        if traj['MMSI'][i] == traj['MMSI'][i-1]:
            if traj['stop/move'][i]==1:
                voyageid.append(0)
                continue
            else:
                if traj['stop/move'][i-1]==1 or time_interval >= 0.083333333:
                    id = id + 1
                    voyageid.append(id)
                    continue
                else:
                    voyageid.append(id)
                    continue
        else:
            id = id + 1
            voyageid.append(id)

    traj['VOYAGEID'] = voyageid
    traj = traj.reset_index(drop=True)

    return traj

def sort_VoyageID(traj):
    ##########################去除停留点，根据长度筛选轨迹#########################
    traj = traj[(traj['stop/move']==0)]
    unique_VOYAGE = traj['VOYAGEID'].value_counts()
    short_VOYAGE = unique_VOYAGE[unique_VOYAGE<500].index.tolist()
    traj = traj[(True^traj['VOYAGEID'].isin(short_VOYAGE))]
    traj = traj.drop(columns=['stop/move'])
    traj = traj.reset_index(drop=True)
    ##########################VOYAGEID重新排序，递增###############################
    voyageid_new = []
    voyageid_new.append(1)
    id = 1
    for i in range(1,len(traj)):
        if traj['VOYAGEID'][i] != traj['VOYAGEID'][i-1]:
            id = id + 1
        voyageid_new.append(id)
    traj['VOYAGEID'] = voyageid_new
    #############################重新变换回地理坐标###############################
    traj = coord_transform(traj,True)

    return traj


if __name__ == '__main__':
    main()





