import numpy as np
import cv2
from matplotlib import pyplot as plt
from cv2CalibMethod import cv2Calibrate

img1 = cv2.imread('mire_1.png')
img2 = cv2.imread('mire_2.png')

corners_1 = cv2.findChessboardCorners(img1,(11,8))
corners_2 = cv2.findChessboardCorners(img2,(11,8))

# cv2.drawChessboardCorners(img1,(11,8),corners_1[1],True)
# cv2.imwrite('./Results/Corners_im1.png',img1)
# cv2.drawChessboardCorners(img2,(11,8),corners_2[1],True)
# cv2.imwrite('./Results/Corners_im2.png',img2)


coord_px = np.concatenate((np.squeeze(corners_1[1]),np.squeeze(corners_2[1])))

#print(coord_px)
#print(coord_px.shape)

# Coord_mm
coord_mm = []
Nx,Ny = 11,8
x,y=0,0
for z in [0,100]:    
    for y in range(Ny):
        for x in range(Nx):
            coord_mm.append(np.array([x*-20,y*-20,z]).astype(np.float32))
coord_mm = np.array(coord_mm)

objp = np.zeros((1, Nx * Ny, 3), np.float32)
objp[0,:,:2] = np.mgrid[0:Nx, 0:Ny].T.reshape(-1, 2) * -20

# Implémentation de TSAI
N = len(coord_mm)
i1,i2 = img1.shape[0]/2,img1.shape[1]/2
#print(i1,i2)
U_tilde = coord_px - np.array([i1,i2])

A = np.zeros((N,7))
U1 = U_tilde[:,0]

for k in range(N):
    A[k,0] = U_tilde[k,1]*coord_mm[k,0]
    A[k,1] = U_tilde[k,1]*coord_mm[k,1]
    A[k,2] = U_tilde[k,1]*coord_mm[k,2]
    A[k,3] = U_tilde[k,1]
    A[k,4] = -U_tilde[k,0]*coord_mm[k,0]
    A[k,5] = -U_tilde[k,0]*coord_mm[k,1]
    A[k,6] = -U_tilde[k,0]*coord_mm[k,2]

# Résoudre AL = U1
L = np.linalg.pinv(A)@U1

# Estimation des paramètres
o2c = 1/np.sqrt(L[4]**2+L[5]**2+L[6]**2)
beta = o2c * np.sqrt(L[0]**2+L[1]**2+L[2]**2)
o1c = (L[3]*o2c)/beta

r11 = L[0]*o2c/beta
r12 = L[1]*o2c/beta
r13 = L[2]*o2c/beta
r21 = L[4]*o2c
r22 = L[5]*o2c
r23 = L[6]*o2c

r31,r32,r33 = np.cross([r11,r12,r13],[r21,r22,r23])

phi = -np.arctan(r23/r33)
gamma = -np.arctan(r12/r11)
omega = np.arctan(r13/(-r23*np.sin(phi)+r33*np.cos(phi)))

# r31 = -np.cos(phi)*np.sin(omega)*np.cos(gamma) + np.sin(phi)*np.sin(gamma)
# r32 = -np.cos(phi)*np.sin(omega)*np.sin(gamma) + np.sin(phi)*np.cos(gamma)
# r33 = np.cos(phi)*np.cos(omega)

# Matrice B et R
B = np.zeros((N,2))
R = np.zeros((N,1))

for k in range(N):
    B[k,0] = U_tilde[k,1]
    B[k,1] = -(r21*coord_mm[k,0] + r22*coord_mm[k,1] + r23 *coord_mm[k,2] + o2c)
    R[k] = -U_tilde[k,1]*(r31*coord_mm[k,0]+r32*coord_mm[k,1]+r33*coord_mm[k,2])

o3c,f2 = np.linalg.pinv(B)@R

# Déterminer s1 et s2
f = 4
s2 = f/f2
s1 = s2/beta
f1 = f2/beta
# print("(s1,s2) = ({},{})".format(s1[0],s2[0]))
# print("(f1,f2) = ({},{})".format(f1[0],f2[0]))

# Taille capteur
capt_x,capt_y = s1*img1.shape[0],s2*img1.shape[1]
#print(capt_x,capt_y)

#print(beta)

# Reprojection des points 
f1 = f/s1[0]
f2 = f/s2[0]
M_int = np.array([[f1,0,i1,0],[0,f2,i2,0],[0,0,0,1]])
M_ext = np.array([[r11,r12,r13,o1c],[r21,r22,r23,o2c],[r31,r32,r33,o3c[0]],[0,0,0,1]])
M = M_int@M_ext
# print(M_ext)

#print(M)
alpha = r31*coord_mm[:,0]+r32*coord_mm[:,1]+r33*coord_mm[:,2]+o3c
U_proj_temp = np.zeros((N,3))
U_proj = np.zeros((N,2))

for k in range(N):
    U_proj_temp[k] = M@np.append(coord_mm[k],1)
    U_proj[k] = U_proj_temp[k,:2]/alpha[k]

# print(U_proj)


# Méthode cv2
M_cam = M_int[:3,:3]
M_cam[2,2] = 1
proj_cv2 = cv2Calibrate(coord_px,coord_mm,M_cam,img1.shape[1],img1.shape[0])

plt.figure()
plt.title('Image 1')
plt.imshow(img1)
plt.scatter(U_proj[:88,0],U_proj[:88,1])
plt.scatter(proj_cv2[0][:88,:,0],proj_cv2[0][:88,:,1])
plt.legend(['Tsau','CV2'])

plt.figure()
plt.title('Image 2')
plt.imshow(img2)
plt.scatter(U_proj[88:,0],U_proj[88:,1])
plt.scatter(proj_cv2[0][88:,:,0],proj_cv2[0][88:,:,1])
plt.legend(['Tsau','CV2'])
plt.show()