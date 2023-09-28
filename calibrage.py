import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('mire_1.png')
img2 = cv2.imread('mire_2.png')

corners_1 = cv2.findChessboardCorners(img1,(11,8))
corners_2 = cv2.findChessboardCorners(img2,(11,8))

# cv2.drawChessboardCorners(img2,(11,8),corners_2[1],True)
# cv2.imshow('img',img2)
# cv2.waitKey(0)

coord_px = np.concatenate((np.squeeze(corners_1[1]),np.squeeze(corners_2[1])))

print(coord_px.shape)

# Coord_mm
coord_mm = []
Nx,Ny = 11,8
x,y=0,0
for z in [0,100]:    
    for y in range(Nx):
        for x in range(Ny):
            coord_mm.append(np.array([x*-20,y*-20,z]))
coord_mm = np.array(coord_mm)
