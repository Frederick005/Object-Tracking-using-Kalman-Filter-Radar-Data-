"""
Created on Sun sep 20 09:47:27 2020

@author: Ananya Frederick
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics

def prediction_State_Vector(x,vel_x,y,vel_y,t,accel):
    '''Prediction step:
    1) X(kp) = AX(k-1) + Bu(k) + w where w equals noise and we are assuming w to be 0

    Predicting the state vector
    '''
    A= np.reshape(np.array([[1,t,0,0],[0,1,0,0],[0,0,1,t],[0,0,0,1]]), (4,4))
    B= np.reshape(np.array([0.5*t**2,t,0.5*t**2,t]), (4,1))
    X_prev= np.reshape(np.array([x,vel_x,y,vel_y]),(4,1))
    X_curr= np.matmul(A,X_prev)+ np.dot(B,accel)
    return (X_curr)

def prediction_Process_Covariance1(cov,t):
    '''2) P(kp) = AP(k-1)A(transpose) + Q where Q is process noise

    Predicting the process covariance
    '''
    A= np.reshape(np.array([[1,t,0,0],[0,1,0,0],[0,0,1,t],[0,0,0,1]]), (4,4))
    G= np.reshape(np.array([0.5*t**2,t,0.5*t**2,t]), (4,1))
    P_curr_inter= np.matmul(A,cov)
    Q = np.dot(accel_var,np.matmul(G,G.T))
    P_curr= np.matmul(P_curr_inter,A.T) + Q
    return (P_curr)

def Kalman_gain(p_curr,meas_cov_curr):
    '''3) KG= P(kp)H(transpose)/HP(kp)H(transpose)
    
    Calculating Kalman_gain
    '''
    H= np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    Num= np.matmul(p_curr,H.T)
    Den_intermediate=np.matmul(H,p_curr)
    Den= np.matmul(Den_intermediate,H.T) + meas_cov_curr
    KG= np.matmul(Num,np.linalg.inv(Den))
    return (KG)

def update_State_vector(KG,X_curr,meas_curr):
    '''4) X= X(kp)+K[y-HX(kp)]
    
    Updating the state vector
    '''
    H= np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    X_updated= X_curr + np.matmul(KG,(meas_curr-np.matmul(H,X_curr)))
    return (X_updated)

def update_Process_Covar(KG, p_curr):
    '''5) P=(I-KH)P(kp)
    
    Updating the process Covariance
    '''
    H= np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    I=np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    ratio= I-np.matmul(KG,H)
    P_updated = np.matmul(p_curr,ratio)
    return(P_updated)

accel_var=24

'''
Predicting for the first time: (Prediction step)                                              #
'''
def predict_point_1 (x,v_x,y,v_y):
    Meas_cov= np.reshape(np.array([[40,0,0,0],[0,0.028,0,0],[0,0,40,0],[0,0,0,0.028]]), (4,4))
    estimate = prediction_State_Vector(x,v_x,y,v_y,0.5,0)
    process_covar = prediction_Process_Covariance1(Meas_cov,0.5)
    return (estimate,process_covar)
'''
After finding the predicted point
1) Calculate the Kalman Gain
2) Update the State vector and Process covariance 
'''
def predict_point_after(estimate,process_covar,Measurements1):
    Meas_cov= np.reshape(np.array([[40,0,0,0],[0,0.028,0,0],[0,0,40,0],[0,0,0,0.028]]), (4,4))

    KG= Kalman_gain(process_covar,Meas_cov)
    identity=np.eye(4)
    KG=KG*identity
    updated_state = update_State_vector(KG,estimate,Measurements1)
    updated_vector = update_Process_Covar(KG,process_covar)
    
    estimate = prediction_State_Vector(updated_state[0][0], updated_state[1][0],updated_state[2][0], updated_state[3][0], 0.5, 0)
    process_covar = prediction_Process_Covariance1(updated_vector,0.5)
    return (estimate,process_covar)