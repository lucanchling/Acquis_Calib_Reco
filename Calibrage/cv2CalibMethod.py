import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def cv2Calibrate(coord_px,coord_mm,M_int,h,w):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (11,8)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = [coord_mm]#[coord_mm[:88,:],coord_mm[88:,:]]
    # objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [coord_px]#[coord_px[:88,:],coord_px[88:,:]] 
    # imgpoints = []
    
    # # # Defining the world coordinates for 3D points
    # objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * -20
    
    # # # Extracting path of individual image stored in a given directory
    # images = glob.glob('./*.png')
    # for fname in images:
    #     img = cv2.imread(fname)
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     # Find the chess board corners
    #     # If desired number of corners are found in the image then ret = true
    #     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    #     objpoints.append(np.squeeze(objp))
    #     imgpoints.append(np.squeeze(corners))
    
        
    # cv2.destroyAllWindows()
    
    # h,w = img.shape[:2]
    
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    # objpoints_bis = np.concatenate(objpoints[0],objpoints[1])
    # imgpoints_bis = np.concatenate(imgpoints[0],imgpoints[1])
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h,w),M_int,None,None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    
    mean_error = 0
    proj = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        proj.append(imgpoints2)
    #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    #     mean_error += error
    # print( "total error: {}".format(mean_error/len(objpoints)) )

    # plt.figure()
    # plt.subplot(211)
    # plt.imshow(images_save[0])
    # plt.scatter(proj[0][:,:,0],proj[0][:,:,1])
    # plt.subplot(212)
    # plt.imshow(images_save[1])
    # plt.scatter(proj[1][:,:,0],proj[1][:,:,1])
    # plt.show()

    return proj