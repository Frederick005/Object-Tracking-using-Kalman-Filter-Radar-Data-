
import os
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import linear_model
from sklearn.cluster import DBSCAN
import math
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from Radartest import radarmain
from pandas import DataFrame

root = '/home/nuscenes/data/sets/nuscenes/'
rmax=80
rmin=0;
alpha =5
beta=3
grid_resolution =2
threshold = 0.75



# remove the ground plane refections using ransac algorithm 

def get_lidar_points(lidar_top,radar_points):
    data_path =os.path.join(root,lidar_top['filename'])
    #print(data_path)
    scan = LidarPointCloud.from_file(data_path)
    
    #transformation
    lidarcalibration = nusc.get('calibrated_sensor',lidar_top_data['calibrated_sensor_token'])
    lidar_trans_matrix=transform_matrix(lidarcalibration['translation'],Quaternion(lidarcalibration['rotation']),inverse=False)
    scan.transform(lidar_trans_matrix)
    lidar_top_data_transformed = scan.points.T
    
    
   # point = scan.reshape((-1, 5))
    inliers,outliers = ransac(lidar_top_data_transformed[:,:3])
    plot_clusters(inliers,outliers)
    ready_for_mapping = remove_non_stationary(outliers,radar_points)
    
    return ready_for_mapping
    
    
#plot ransac output cluster 
def plot_clusters(inliers,outliers):
    cluster_outliers = dbscan(outliers)
    print ('Displaying cluster of outliers for this sample.....')
    show_clusters(outliers, cluster_outliers)
    cluster_inliers = dbscan(inliers)
    print('Displaying cluster of inliers for this sample.....')
    show_clusters(inliers,cluster_inliers)

# using classified moving points from radar to remove the non staitonary lidar points of objects  

def remove_non_stationary(outliers,radar_points):
    x_r = radar_points[:,0]
    y_r = radar_points[:,1]
    ind_x = []
    ind_y = []
    for i in range (len(radar_points)):
        x_low = x_r[i]-0.3  
        x_high = x_r[i]+0.3
        y_low = y_r[i]-0.3
        y_high = y_r[i]+0.3
        for j in range (len(outliers)):
            if outliers[j][0]<x_high and outliers[j][0]>x_low:#checking for lidar points which lies within the bounded
                ind_x.append(j)                               #range of same x and y distance from vehicle as of radar moving detections
            if outliers[j][1]<y_high and outliers[j][1]>y_low:
                ind_y.append(j)
    indices=np.intersect1d(ind_x,ind_y) #Extracting the similar indices from x & y indexes
    final_outliers = np.delete(outliers, indices, 0) #deleting the rows of moving detection points from Lidar
    return final_outliers


#extract the x, y, z points from lidar data 
def get_lidar_data(lidar_points):
    x_values = lidar_points[:,0]
    y_values = lidar_points[:,1]
    z_values = lidar_points[:,2]
    return x_values,y_values,z_values



#create a grid based on the available x and y values for each sample.
def make_grid(x_values,y_values):
    max_x_value_width= max(x_values)

    min_x_value_width=min(x_values)

    length_of_x = np.int32((max_x_value_width - min_x_value_width)/(grid_resolution))
    #print(length_of_x)

    max_y_value_length= max(y_values)
    min_y_value_length=min(y_values)

    length_of_y = np.int32((max_y_value_length - min_y_value_length)/(grid_resolution))
    #print(length_of_y)

    m=np.multiply(0.5,np.ones((length_of_x,length_of_y)))
    return m



def inverse_sensor_model(length_of_x,length_of_y,meas_r,meas_phi):
    m1 = np.zeros((length_of_x, length_of_y))
    for i in range(length_of_x):
        for j in range(length_of_y):
            
            r = np.round((math.sqrt((i -(length_of_x/2))**2 + (j -(length_of_y/2))**2)),decimals =1)
            phi = np.round((math.atan2((i-(length_of_x/2)),(j-(length_of_y/2)))),decimals =1)
            
           
            k = np.argmin(np.abs(np.subtract(phi, meas_phi)))
           
           
        
            if (r > min(rmax, meas_r[k] + alpha /2.0)) or (abs(phi - meas_phi[k]) > beta/2.0):
                m1[i, j] = 0.5
                continue
            # If the range measurement lied within this cell, it is likely to be an object.
            if ((meas_r[k] < rmax) and (abs(r - meas_r[k]) < alpha / 2.0)):
                m1[i, j] = 0.7
                continue
            # If the cell is in front of the range measurement, it is likely to be empty.
            if r < meas_r[k]:
                m1[i, j] = 0.3
            
    return m1
            
    


# Removal of Ground plane reflection points using Ransac algorithm 
def ransac(points):   
    
    xy = points[:, :2]
    z = points [:,2]
    #print (xy)
    #print(z)
    ransac = linear_model.RANSACRegressor(residual_threshold = 0.2)
    ransac.fit(xy, z)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    inliers = np.zeros(shape=(len(inlier_mask), 3))
    outliers = np.zeros(shape=(len(outlier_mask), 3))
    a,b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_
    k={}
    for i in range(len(inlier_mask)):
        if not outlier_mask[i]:
            inliers[i] = points[i]
        
        else:
            outliers[i] = points[i]
        

    ind_outliers=[]
    ind_inliers=[]
    for i in range(len(outliers)):
        if outliers[i][0]==0 and outliers[i][1]==0 and outliers[i][2]==0:
            ind_outliers.append(i)
        if inliers[i][0]==0 and inliers[i][1]==0 and inliers[i][2]==0:
            ind_inliers.append(i)
           
    outliers = np.delete(outliers,ind_outliers,0)
    inliers = np.delete(inliers,ind_inliers,0)
    return inliers,outliers

#clustering of lidar points using dbscan
def dbscan(array_in):
    clustering = DBSCAN(eps=6,min_samples = 4).fit(array_in)
    clustering =clustering.labels_
    return clustering

#Display of clusters
def show_clusters(array,cluster,flag=0):
    df = DataFrame(dict(x=array[:,0],y=array[:,1],label=cluster))  
    unique_labels = set(cluster)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 2, len(unique_labels))]         
             
    fig,ax = plt.subplots(figsize=(7,7))
    
    grouped = df.groupby('label')    
    for key,group in grouped:        
        group.plot(ax=ax, kind = 'scatter', x='y',y='x', color = 'Green')
    plt.ylabel('Distance along x')    
    plt.xlabel('Distance along y') 

    plt.xlim(60,-60)
    plt.ylim(-40,70)
    plt.grid(True)           
    plt.show()          
    df.groupby('label')


#------------End of definitions------------------------------------------

nusc = NuScenes(version='v1.0-mini', dataroot='/home/nuscenes/data/sets/nuscenes', verbose=False)
scene_no = -1

while scene_no < 0 :
    try:
        scene_no= int(input("Please enter the Scene number : "))
    except ValueError:
        print ("This is not a number! Please try again.")
        
no_of_iterations = -1
while no_of_iterations < 0 :
    try:
        no_of_iterations= int(input("Please enter the number of samples required : "))
    except ValueError:
        print ("This is not a number! Please try again.")
        
my_scene = nusc.scene[scene_no]

iteration =0                                          # For nimber of samples
sample_token = my_scene['first_sample_token']         # get first sample token from nuscenes 
my_sample_current = nusc.get('sample', sample_token) 
m3 =[]                                                #Array to append the occupancy grid values for each sample
m_grid =[]                                            

while(iteration < no_of_iterations):
    
    print('Processing sample No.{}........'.format(iteration))
    
    lidar_top_data = nusc.get('sample_data',my_sample_current['data']['LIDAR_TOP'])
    
    radar_points= radarmain(nusc, sample_token, iteration) # Get processed radar points for non stationary objects
    
    lidar_points = get_lidar_points(lidar_top_data,radar_points) # get raw lidar data from nuscenes
    
    x ,y ,z = get_lidar_data(lidar_points) # get processed lidar data after all filters 
                                           #(removed ground point reflections,non stationary and above lidar height points)

                                           
    next_token = my_sample_current['next']  #get the next token and sample using current sample 
    next_sample= nusc.get('sample', next_token) 
    
    sample_token = next_token
    my_sample_current = next_sample

    m= make_grid(x,y)                        #Make a imaginary grid according the samples received 
    L0 =np.log(np.divide(m,np.subtract(1,m))) # get the initial logit probabilities
    
    #calculate max of x and y values based sample data 
    max_x_value_width= max(x)
    min_x_value_width=min(x)
    length_of_x = np.int32((max_x_value_width - min_x_value_width)/(grid_resolution))
  
    max_y_value_length= max(y)
    min_y_value_length=min(y)
    length_of_y = np.int32((max_y_value_length - min_y_value_length)/(grid_resolution))

    # measure the distance and angle of each measurements 
    meas_r = np.hypot(x,y) 
    meas_phi = np.arctan2(x,y)
      

    #apply inverse sensor model
    inverse_model_output =inverse_sensor_model(length_of_x,length_of_y,meas_r,meas_phi)
    
    # Put result of inverse sensor model in log -odd form 
    L =np.log(np.divide(inverse_model_output,np.subtract(1,inverse_model_output)))-L0

    #convert log odds to  real probabilities
    m2=np.exp(L)/(1+np.exp(L))

    #move to next sample in the scene123
    iteration = iteration +1 
    m3.append(m2)
    
    print('********** End of Sample {} **********'.format(iteration))
    print('\n')
    
# print the occupancy grid 4 samples in a scene    
plt.figure()
f,axarr = plt.subplots(no_of_iterations,1)
f.set_size_inches(18,15)
for index in range(no_of_iterations):
    axarr[index].imshow(m3[index],cmap='Greys')
    
print('All requested Lidar and Radar Sample data of Scene No.{} have been displayed!'.format(scene_no))
print('********** End of Session **********')

# Enable to render images for comparision 
    
i=0
sample_token = my_scene['first_sample_token']    
my_sample_current_camera = nusc.get('sample', sample_token)
while(i<no_of_iterations):
    camera_front_data = nusc.get('sample_data',my_sample_current_camera['data']['CAM_FRONT'])
    nusc.render_sample_data(camera_front_data['token'])

    next_token = my_sample_current_camera['next']
    #print(next_token)
    next_sample= nusc.get('sample', next_token)
    #print(next_sample)

    sample_token = next_token
    my_sample_current_camera = next_sample
    i=i+1


