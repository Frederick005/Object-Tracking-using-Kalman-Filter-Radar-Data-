"""
@author: Justin
FH Dortmund

Use this class to get the values of the radar detections. See descriptions of 
the classmethods to get useful informations.


################################### EXAMPLE ###################################

(See nuScenes tutorial to find out how to get the path easily)
...
...
radar_points = radar_processing_utils.Radar.get_radar_points(path_of_radar_pcd)
distance, phi, radial_velocity = radar_processing_utils.Radar.get_radar_data(radar_points)
...
# Continue with processing

###############################################################################

"""
from nuscenes.utils.data_classes import RadarPointCloud
import numpy as np

class Radar:
    
    
    def __init__(self):
        super().__init__()
        
        
    @classmethod
    def get_radar_points(cls, data_path) -> 'RadarPointCloud':
        """
        This method loads Radar data from a PointCloudData-File. You need 
        give the path of the PointCloud-File from '/data/sets/nuscenes/samples/RADAR..'.

        Parameters
        ----------
        data_path : string
            The path of the PointCloud-File from '/data/sets/nuscenes/samples/RADAR..'.
            See nuScenes tutorial to find out how to get the path easily.

        Returns
        -------
        radar_points : np.ndarray(18, n)
            Point cloud matrix with 18 dimensions and n (<= 125) points. Use
            this array/matrix in 'calculation_of_radar_data' to get the values
            of the radar measurements.

        """
        
        # To get all points/detections we need to disable settings (..._states)
        #RadarPointCloud.disable_filters()
        radar_pcls = RadarPointCloud.from_file(data_path,
                                               invalid_states = list(range(18)),
                                               dynprop_states = list(range(8)),
                                               ambig_states = list(range(5)))
        
        radar_points = radar_pcls.points
        
        return radar_points
    
    
    
    @classmethod
    def get_radar_data(cls, radar_points) -> np.ndarray:
        """
        This method calculates the raw radardata (distance, phi, radial-velocity)
        of every point/detection out of the PointCloud.

        Parameters
        ----------
        radar_pcls : np.ndarray
            PointCloud (RadarPointCloud-Object) from 'get_radar_points'.

        Returns
        -------
        detections_distance : np.ndarray
            An n*1-array with the distances of the radar-detections to the radarsensor (n <= 125).
        detections_phi : np.ndarray
            An n*1-array with the angles of the radar-detections r.t. the radarsensor (n <= 125).
        detections_radial_velocity : np.ndarray
            An n*1-array with the radial velocities of the radar-detections (n <= 125).
            Note that these velocities not consider the velocity of the ego-vehicle.

        """
        x_array = radar_points[0]
        y_array = radar_points[1]
        
        vx_array = radar_points[6]        
        vy_array = radar_points[7]
        
        # calculate the distances of the detections to the radar-sensor
        detections_distance = np.sqrt(x_array**2 + y_array**2)
   
        # calculate the angle (phi, degrees) of the detections w.r.t the radar-sensor-frame
        detections_phi = np.rad2deg( np.arctan( y_array / x_array ))
        
        # calculate the radial velocity of the detections to the radar-sensor
        # w.r.t detections_phi (!!!)
        detections_radial_velocity = vx_array * np.cos(np.deg2rad(detections_phi))
        vx = vx_array * np.cos(np.deg2rad(detections_phi))
        vy = vy_array * np.sin(np.deg2rad(detections_phi))
        
        return detections_distance, detections_phi, vx, vy
