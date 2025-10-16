# Object Tracking using Kalman Filter Radar Data

## Introduction
Built and integrated a from-scratch 4D Kalman Filter in Python to track moving objects in urban driving scenes using nuScenes radar/LiDAR data. The pipeline clusters radar detections, compensates for ego-motion, associates clustered centroids across frames, and delivers stable tracks. My primary contribution was the Kalman Filter implementation and its end-to-end integration into the radar processing workflow, using real nuScenes data and practical data association techniques

## Project Overview
The project involved the following key components:

**Kalman Filter Implementation**: I wrote the code for a Kalman filter from scratch in Python, which included the prediction and update steps.
**Integration with Main Code**: I merged the Kalman filter code with the main project code, allowing the filter to utilize car location details and track the vehicle.
**Data Processing**: The project involved processing data from various sensors, including radar and lidar, to provide accurate location tracking.
**Adaptive Kalman Filtering**: I implemented an adaptive Kalman filtering algorithm to improve the tracking accuracy in cases where the vehicle's maneuver is unpredictable.

## Skills Gained
Through this project, I gained the following skills:

**Python Programming**: I improved my proficiency in Python, including data structures, object-oriented programming, and file management.
**Kalman Filter Algorithm**: I gained a deep understanding of the Kalman filter algorithm, including its mathematical foundations and implementation details.
**Adaptive Kalman Filtering**: I learned about the concept of adaptive Kalman filtering and implemented an algorithm to improve the tracking accuracy.
**Data Processing and Analysis**: I learned to process and analyze large datasets from various sensors, including radar and lidar.
**Problem-Solving and Debugging**: I developed strong problem-solving and debugging skills, allowing me to identify and resolve issues in the code.

## Recommendations for Future Work
**Optimization**: The Kalman filter and adaptive Kalman filtering algorithms can be optimized for better performance and accuracy.
**Extension to Other Sensors**: The project can be extended to include other sensors, such as cameras and GPS, to provide a more comprehensive tracking system.
**Application to Other Domains**: The project can be applied to other domains, such as robotics and surveillance, where accurate tracking and localization are critical.

## Conclusion
In conclusion, this project successfully implemented a Kalman filter and adaptive Kalman filtering algorithms to track vehicle locations using sensor data. The use of adaptive Kalman filtering improved the tracking accuracy, particularly in cases where the vehicle's maneuver was unpredictable. The project showcased the effectiveness of combining complex algorithms with real-world data to achieve accurate tracking and localization. Overall, the project demonstrated the potential of Kalman filtering and adaptive Kalman filtering in various applications, including autonomous vehicles and sensor fusion
