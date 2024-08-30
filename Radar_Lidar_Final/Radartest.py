#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 00:47:27 2020

@author: Naveen Chengappa
"""

from radar_processing_utils import Radar as rp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from pandas import DataFrame
import numpy as np
import os
from copy import copy

# jupyter notebook /home/nuscenes/nuscenes-devkit/python-sdk/tutorials/nuscenes_basics_tutorial.ipynb

sensor = 'RADAR_FRONT'
sensor2 = 'CAM_FRONT'
root = '/home/nuscenes/data/sets/nuscenes/'

threshold = 0.8
array_in = np.empty([125,2])


""" 
    detection_points(mypts)
    This function sorts the x and y axis points to a single global array
"""
def detection_points(mypts):
    x_array = mypts[0]
    y_array = mypts[1]   
    
    for i in range(len(x_array)):
        array_in[i][0] = x_array[i]
        array_in[i][1] = y_array[i] 
        
        
""" 
    get_box(cluster,array_cluster,flag)
    This function makes boxes around the clustered points
"""        
def get_box(cluster,array_cluster,flag):
    cluster_num = len(set(cluster))-1    
    rect = []
    width = 0
    height = 0
    
    for i in range(cluster_num): 
        box_x = []
        box_y = []
        for k, cluster_pt in enumerate(cluster):            
            if cluster_pt == i:                
                box_y.append(array_cluster[k][0])
                box_x.append(array_cluster[k][1])              

        width = max(box_x) - min(box_x)
        height = max(box_y) - min(box_y)        
        #if flag == 1:
            #rect.append(Rectangle((min(box_x),min(box_y)), width, height, fill=False, color='blue'))
        if flag == 2:
            rect.append(Rectangle((min(box_x),min(box_y)), width, height, fill=False, color='green'))
        if flag == 3:            
            rect.append(Rectangle((min(box_x),min(box_y)), width, height, fill=False, color='red'))     
        
    return rect  
    
""" 
    dbscan(array_in,flag)
    This function implements the DBscan algorithm on the set of radar points.
    eps : epsilon is taken as 3.5
    min_samples : minimum number of samples in a cluster is approximated as 4
    returns : clusters in labels
"""         
def dbscan(array_in,flag): 

    if array_in.size != 0:            
        if flag == 1:
           clustering = DBSCAN(eps=3.5,min_samples=4).fit(array_in)
        if flag == 2:
           clustering = DBSCAN(eps=3.5,min_samples=2).fit(array_in) 
        if flag == 3:
           clustering = DBSCAN(eps=3.5,min_samples=2).fit(array_in)        

        cluster = clustering.labels_
        return cluster
    else:
        return np.empty(shape = array_in.shape)  

""" 
    show_clusters(array_cluster,cluster,frameNo,flag=0)
    This function displays the clusters in different colours and
    also adds the box around the clusters.
"""   
def show_clusters(array_cluster,cluster,frameNo,flag=0):
  
   
    df = DataFrame(dict(x=array_cluster[:,0],y=array_cluster[:,1],label=cluster))   
    unique_labels = set(cluster)    
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]  
      
    fig,ax = plt.subplots(figsize=(8,8))
    plt.ylabel('Longitude (x, in Meter)')    
    plt.xlabel('Latitude (y, in Meter)') 
    if flag == 0:
        plt.title(' Null')
    if flag == 1:
        plt.title(' All Detection Points')
    if flag == 2:
        plt.title(' Frame {} - Moving points'.format(frameNo))
    if flag == 3:
        plt.title(' Frame {} - Stationary points'.format(frameNo))        
    plt.xlim(30,-40)
    plt.ylim(0,120)
            
    grouped = df.groupby('label')    
    for key,group in grouped:        
        group.plot(ax=ax, kind = 'scatter', x='y',y='x', label=key, color = colors[key])
    
    rect = get_box(cluster,array_cluster,flag)    
    for c in rect:
        new_c=copy(c)
        ax.add_patch(new_c)    

    plt.grid(True)  
    plt.legend(loc = 'upper right')
  
    # if flag == 2:        
    #     plt.savefig('Moving_{}.png'.format(frameNo), dpi=300, bbox_inches='tight') 
    # if flag == 3:        
    #     plt.savefig('Ambig_{}.png'.format(frameNo), dpi=300, bbox_inches='tight')   
    plt.show() 
    df.groupby('label')   

""" 
    get_ego_speed(nusc, sample_frame)
    This function gets the speed of the ego vehicle in the current frame.
"""         
def get_ego_speed(nusc, sample_frame):    
           
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
    #if speed < 1:
        #print('The speed of ego vehicle is : 0 m/s')
    #else:
        #print('The speed of ego vehicle is : {:.2f} m/s'.format(speed))
    
    return speed  

""" 
    moving_pts(move, ambig, array_in)
    This function provides an array of moving and stationary points
    from the set of all detected points.
    returns : 2 arrays of moving and stationary points
"""
def moving_pts(move, ambig, array_in):
    array_move = np.empty([125,2])
    array_ambig = np.empty([125,2])
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
            
    array_move_dup = np.copy(array_move)
    array_ambig_dup = np.copy(array_ambig)
    moving_points = np.delete(array_move_dup,index_move,0)
    ambig_points = np.delete(array_ambig_dup,index_ambig,0)
    
    return moving_points,ambig_points

""" 
    classify_target(v_comp,cluster)
    This function will classify points as moving or ambiguous 
    depending on a threshold value defined.
    returns : 2 arrays of moving and stationary points
"""            
def classify_target(v_comp,cluster):
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

""" 
    get_radar_data(radar_front,spd, frameNo)
    This function fetches the radar data , radial velocity, phi and distance
    from the nuscenes dataset.    
"""
def get_radar_data(radar_front,spd, frameNo):
    dist = []    
    phi = []    
    velo = []    
    
    data_path = os.path.join(root, radar_front['filename'])
    mypts = rp.get_radar_points(data_path)
    radar_data = rp.get_radar_data(mypts)    
    
    detection_points(mypts)    
    cluster = dbscan(array_in,flag=1)
    #show_clusters(array_in,cluster,flag=1)
    
    for i in range(len(radar_data[0])):
        dist.append(radar_data[0][i])
        phi.append(radar_data[1][i])
        velo.append(radar_data[2][i])            
     
    rad_spd = spd * np.array(np.cos(np.deg2rad(phi)))
    v_comp2 = rad_spd + np.array(velo)
    
    move2,ambig2 = classify_target(v_comp2, cluster)
    
    array_move2,array_ambig2 = moving_pts(move2, ambig2, array_in)
    if array_move2.size != 0:
        cluster_move2 = dbscan(array_move2,flag=2)
        print('Generating Radar detected moving objects of sample {} ..... '.format(frameNo))
        show_clusters(array_move2,cluster_move2,frameNo,flag=2)
    else:
        print('No moving points detected in Frame {}'.format(frameNo))
        
    if array_ambig2.size != 0:    
        cluster_ambig2 = dbscan(array_ambig2,flag=3)
        show_clusters(array_ambig2,cluster_ambig2,frameNo,flag=3)
    else:
        print('No ambiguous points detected in Frame {}'.format(frameNo))
        
    return array_move2        

#-----------------------------------------------------------------------------#        
#-----------------------------------------------------------------------------#
""" 
    radarmain(nusc, sample_token, iteration)
    This function is the main Radar function which invokes all the other fucntions
    to provide the radar view.
    It also returns the array of moving points as captured in one sample of the scene.        
"""
def radarmain(nusc, sample_token, iteration):
	my_sample = nusc.get('sample', sample_token)
	radar_front_data = nusc.get('sample_data',my_sample['data'][sensor])
    
	spd = get_ego_speed(nusc, my_sample)
	array_move2 = get_radar_data(radar_front_data,spd, iteration)  

	return array_move2
#-----------------------------------------------------------------------------#  

'Use this code to run this file independently'    
# nusc = NuScenes(version='v1.0-mini', dataroot='/home/nuscenes/data/sets/nuscenes', verbose=False)
# my_scene = nusc.scene[3]

# sample_token = my_scene['first_sample_token'] 

# movzz = radarmain(nusc, sample_token, 0)    
