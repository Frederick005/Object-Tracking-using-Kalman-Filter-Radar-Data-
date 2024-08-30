#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 00:47:27 2020

@author: Naveen Chengappa
"""

from nuscenes.nuscenes import NuScenes
from radar_processing_utils import Radar as rp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from pandas import DataFrame
import numpy as np
import os
from nuscenes.utils.data_classes import RadarPointCloud
from Kalman_Filter_4D_implementation import *
import math
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import cv2
import os.path as osp

# jupyter notebook /home/nuscenes/nuscenes-devkit/python-sdk/tutorials/nuscenes_basics_tutorial.ipynb

sensor = 'RADAR_FRONT'
sensor2 = 'CAM_FRONT'
root = 'C:/Users/Student/Downloads/SICO/project/data/set/nuscenes'

threshold = 0.75
array_in = np.empty([125,2])

def dbscan(array_in):     
    dummy = np.empty(shape = array_in.shape)    
    if array_in.size != 0:        
        clustering = DBSCAN(eps=3.5,min_samples=2).fit(array_in) 
        cluster = clustering.labels_
        return cluster
    else:
        return dummy      
   
def global_pt_cld(array_cluster,cluster,velo,flag=0):
    centroid = []
    centroid_velo = []     
    
    df = DataFrame(dict(x=array_cluster[:,0],y=array_cluster[:,1],vx = velo[:,0], vy = velo[:,1],label=cluster))      
    df.drop(df[df.label < 0].index, inplace=True)
    grouped = df.groupby('label')
    for key,group in grouped:  
        #print(group)                
        centroid.append([np.mean(group.x),np.mean(group.y)])  
        centroid_velo.append([np.mean(group.vx),np.mean(group.vy)])           
           
    unique_labels = set(cluster)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels)-1)] 

    plt.figure()
    plt.imshow(map_, cmap='gray', vmin=0, vmax=255)
    for my_x,my_y in centroid:
        x, y = basic_mask.to_pixel_coords(my_x , my_y )        
        plt.scatter(x,y)
        
    #print(centroid_velo)

    return centroid, centroid_velo            
        
       
def get_ego_speed(sample_frame):    
           
    next_tkn = sample_frame['next']
    sample_frame2 = nusc.get('sample', next_tkn)
    radar_front_data = nusc.get('sample_data',sample_frame['data'][sensor]) 
    radar_front_data2 = nusc.get('sample_data',sample_frame2['data'][sensor])
    
    ego_pose_record_1 = nusc.get('ego_pose', radar_front_data['ego_pose_token'])
    ego_pose_record_2 = nusc.get('ego_pose', radar_front_data2['ego_pose_token'])    
    
    distance = np.linalg.norm(ego_pose_record_1['translation']) - np.linalg.norm(ego_pose_record_2['translation'])
    time = 1e-6 * (ego_pose_record_2['timestamp'] - ego_pose_record_1['timestamp'])
    #print('Time is : {} seconds'.format(time))
    #print('Distance is : {} meters'.format(distance))    
    speed = abs(distance)/time
    if speed < 1:
        print('The speed of ego vehicle is : 0 m/s')
    else:
        print('The speed of ego vehicle is : {:.2f} m/s'.format(speed))
    
    return speed    

def moving_pts(move, ambig, velo, array_in):
    array_move = np.empty([125,2])
    array_ambig = np.empty([125,2])
    array_velo = np.empty([125,2])
    
    index_ambig = []
    index_move = []   
    
    for i in range(125):
        
        for k in range(2):
            array_move[i][k]=0            
            array_ambig[i][k]=0             
    
    for i in range(len(move)):
        if move[i] != 0:            
            array_move[i][0] = array_in[i][0]
            array_move[i][1] = array_in[i][1]            
            
        elif ambig[i] != 0:            
            array_ambig[i][0] = array_in[i][0]
            array_ambig[i][1] = array_in[i][1]
            
    for i in range(len(move)):        
        if array_move[i][0] ==0 and array_move[i][1] ==0:
            index_move.append(i)           
        if array_ambig[i][0] == 0 and array_ambig[i][1] == 0:
            index_ambig.append(i)
            
    array_velo = np.copy(velo)       
            
    array_move_dup = np.copy(array_move)
    array_ambig_dup = np.copy(array_ambig)    
    
    moving_points = np.delete(array_move_dup,index_move,0)
    ambig_points = np.delete(array_ambig_dup,index_ambig,0)
    moving_velo = np.delete(array_velo,index_move,0)
    
    return moving_points,ambig_points, moving_velo
            
def classify_target(v_comp):
    moving = []
    ambigous = []
    for i in range(len(v_comp)):                    
        if (v_comp[i] > threshold) or (v_comp[i] < -threshold) :
            moving.append(v_comp[i])
            ambigous.append(0)
        else:
            moving.append(0)
            ambigous.append(v_comp[i])
    
    return np.array(moving),np.array(ambigous)

def get_radar_data(radar_front, global_from_sensor):
    dist = []    
    phi = []    
    velo_x = []   
    velo_y = []
    global track_id
    global end_track
    distance=[]
    
    data_path = os.path.join(root, radar_front['filename'])  
    pc = RadarPointCloud.from_file(data_path, invalid_states = list(range(18)),
                                               dynprop_states = list(range(8)),
                                               ambig_states = list(range(5)))    
    
    # Transform pointcloud    
    pc.transform(global_from_sensor)     
    radar_points = pc.points     
    radar_data = rp.get_radar_data(radar_points) 
     
    for i in range(len(radar_data[0])):
        dist.append(radar_data[0][i])
        phi.append(radar_data[1][i])    
        velo_x.append(radar_data[2][i]) 
        velo_y.append(radar_data[3][i])
    
    for i in range(len(pc.points[0])):
        array_in[i][0] = radar_points[0][i]
        array_in[i][1] = radar_points[1][i]    
     
    rad_spd = spd * np.array(np.cos(np.deg2rad(phi)))
    v_comp2 = rad_spd + np.array(velo_x)
    
    target_velo_x = np.array(velo_x) / np.array(np.cos(np.deg2rad(phi)))
    target_velo_y = np.array(velo_y) / np.array(np.sin(np.deg2rad(phi)))
    
    target_velo = np.vstack((target_velo_x, target_velo_y)).T
    
    move2,ambig2 = classify_target(v_comp2)
    
    array_move2,array_ambig2, array_velo = moving_pts(move2, ambig2, target_velo, array_in)
    num=0
    if array_move2.size != 0:
         cluster_move2 = dbscan(array_move2) 
         '''
         1) Calculating the mean position of the cluster point
         2) Calculating the Average velocity for the cluster of points
         '''
         centroid_arr, centroid_velo_arr = global_pt_cld(array_move2,cluster_move2, array_velo,flag=2)
         '''
         #Implementing Single Point Tracking using Kalman Filter                                                                  #
         ##########################################################################################################################
         #Prediction step: Using the Kalman Filter to predict the point in the next frame
         @ Ananya Frederick
         '''
         if track_id==0 and end_track==0:
             for c in centroid_arr:
                 print(c)
             estimate, process_covar= predict_point_1(centroid_arr[num][0], centroid_velo_arr[num][0], centroid_arr[num][1],centroid_velo_arr[num][1])
             estimate_store.append(estimate)
             process_covar_store.append(process_covar)
             track_store.append(centroid_arr[num])
             print(estimate_store)
             track_id=track_id+1
         ##########################################################################################################################
         #Association and Updation Step                                                                                           #
         ##########################################################################################################################
         elif end_track==0:
             for c in centroid_arr:
                 distance.append(math.sqrt(pow((estimate_store[track_id-1][0]-c[0]),2)+pow((estimate_store[track_id-1][2]-c[1]),2)))
             d= min(distance)
             '''
             #Gating
             #Assign a particular area where we predict the predicted point to be
             '''
             if d<=13:
                 pos= distance.index(d)
                 track_store.append(centroid_arr[pos])
                 estimate,process_covar = predict_point_after(np.array(estimate_store[track_id-1]),np.array(process_covar_store[track_id-1]),np.reshape(np.array([centroid_arr[pos][0],centroid_velo_arr[pos][0], centroid_arr[pos][1],centroid_velo_arr[pos][1]]),(4,1)))
                 estimate_store.append(estimate)
                 print(np.array(process_covar_store[track_id-1]))
                 process_covar_store.append(process_covar)
                 track_id=track_id+1
             else :
                 print("End Track")
                 end_track=1
    else:
         print('No moving points detected in Frame')          


def plot_tracks(track_store,map_):
    plt.figure()
    x_=[]
    y_=[]
    for my_x,my_y in track_store:
        x, y = basic_mask.to_pixel_coords(my_x , my_y )     
        x_.append(x)
        y_.append(y)
    plt.plot(x_,y_)
    plt.scatter(x_,y_)
    plt.imshow(map_, cmap='gray', vmin=0, vmax=255)
    plt.show()     
    
#-----------------------------------------------------------------------------#        
#-----------------------------------------------------------------------------#
nusc = NuScenes(version='v1.0-mini', dataroot='C:/Users/Student/Downloads/SICO/project/data/set/nuscenes', verbose=False)
my_scene = nusc.scene[0]

count = 0
track_id = 0
end_track =0
track_store=[]
estimate_store = []
process_covar_store=[]
sample_token = my_scene['first_sample_token']
#current_sample = nusc.get('sample', sample_token)
#sample_token=current_sample['next']

'Change count limit to the number of samples required'
'If all samples of the scene are required then enable the commented statement'

while(count < 10):
       
    my_sample = nusc.get('sample', sample_token)
    
    scene_token = my_sample['scene_token']
    log_token = nusc.get('log', nusc.get('scene', scene_token)['log_token'])    
    radar_front_data = nusc.get('sample_data',my_sample['data'][sensor])    
    ego_pose = nusc.get('ego_pose', radar_front_data['ego_pose_token'])
    cal_sensor = nusc.get('calibrated_sensor', radar_front_data['calibrated_sensor_token'])
    basic_mask = nusc.get('map', log_token['map_token'])['mask']    
    car_from_senor = transform_matrix(cal_sensor['translation'], Quaternion(cal_sensor['rotation']), inverse=False)
    global_from_car = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
    
    # Combine tranformation matrices
    global_from_sensor = np.dot(global_from_car, car_from_senor)
    
    spd = get_ego_speed(my_sample)   
      
    filename = nusc.get('map', log_token['map_token'])['filename']
    map_ = cv2.imread(osp.join(nusc.dataroot, filename), cv2.IMREAD_GRAYSCALE)    
    get_radar_data(radar_front_data, global_from_sensor) 

    sample_token = my_sample['next']    
    count = count + 1 
    
plot_tracks(track_store,map_)
    
      