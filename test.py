import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load previously saved data
with np.load('R1.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]



def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
def draw2(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

img1 = cv.imread('fotos/box1.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('fotos/cenario3.jpg') # trainImage
img2Gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2Gray,None)
pts = cv.KeyPoint_convert(kp1)
pts3d = np.insert(pts, 2, 1, axis=1)
print("kp1.x")
print(len(pts))

pts2d = cv.KeyPoint_convert(kp2)


# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)


print(pts3d[matches[1].imgIdx])

print(pts2d[matches[1].queryIdx])

a = []
b = []
for i in range(len(matches)):
    a.append(matches[i].trainIdx)
    b.append(matches[i].queryIdx)
    print(matches[i].distance)
print("b:")
print(b)

pts2dd = np.asarray(pts2d) 
pts3dd = np.asarray(pts3d) 

pts2dd = pts2dd[a]
pts3dd = pts3dd[b]


# Draw first 10 matches.
# img3 = cv.drawMatches(img1,kp1,img2Gray,kp2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)




# ret, corners = cv2.findChessboardCorners(gray, (8,8),None)
if True == True:
    # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # print(corners2)
    # print(len(corners2))
    # Find the rotation and translation vectors.
    print(len(pts3dd))
    print(len(pts2dd))
    print(pts3dd)
    print(pts2dd)
     

    ret,rvecs, tvecs = cv.solvePnP(pts3dd, pts2dd, mtx, dist)
    
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(20*axis, rvecs, tvecs, mtx, dist)
    print("imgpts")
    print(imgpts)
    img2 = draw2(img2,pts2d,imgpts)
    cv.imshow('img',img2)
    k = cv.waitKey(0) & 0xFF    
# plt.imshow(img),plt.show()
