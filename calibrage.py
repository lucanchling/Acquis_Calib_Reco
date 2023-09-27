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
coord_mm = np.zeros(176,3)
Nx,Ny = 11,8
x,y=0,0
for i in range(88):
    coord_mm[i] = np.array([x*20,y*20,0])
    coord_mm[i+88] = np.array([x*20,y*20,100])