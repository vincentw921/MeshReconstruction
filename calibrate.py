#
# To run:
#
# Install opencv modules in Anaconda environment:
#
#   conda install opencv
#
# Run calibrate.py from the commandline:
#
#   python calibrate.py

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from camutils import Camera

# file names, modify as necessary
calibimgfiles_C0 = "/Users/vincentwang/Desktop/CS117/Final Project/calib/frame_C0_*_u.png"
calibimgfiles_C1 = "/Users/vincentwang/Desktop/CS117/Final Project/calib/frame_C1_*_u.png"
resultfile = 'calibration.pickle'

# checkerboard coordinates in 3D
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = 2.8*np.mgrid[0:8, 0:6].T.reshape(-1,2)

fid = open(resultfile, "wb" )

cameras = []
Ks = []
dists = []

for calibimgfiles in [calibimgfiles_C0, calibimgfiles_C1]:
    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibimgfiles)

    if len(images)==0:
        print('No images found')
        exit()

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Display image with the corners overlayed
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # now perform the calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    R_obj_to_cam, _ = cv2.Rodrigues(rvecs[0])
    t_obj_to_cam = tvecs[0]

    R = R_obj_to_cam.T
    t = -R_obj_to_cam.T @ t_obj_to_cam

    print("Estimated camera intrinsic parameter matrix K")
    print(K)
    print("Estimated radial distortion coefficients")
    print(dist)

    print("Individual intrinsic parameters")
    print("fx = ",K[0][0])
    print("fy = ",K[1][1])
    print("cx = ",K[0][2])
    print("cy = ",K[1][2])
    
    print("Estimated extrinsic parameters for the first image")
    print("Rotation matrix R")
    print(R)
    print("Translation vector t")
    print(t)
    
    # Create camera object
    cam = Camera(
        f=np.array([[K[0, 0]], [K[1, 1]]]),
        c=np.array([[K[0, 2]], [K[1, 2]]]),
        R=R,
        t=t
    )
    
    cameras.append(cam)
    Ks.append(K)
    dists.append(dist)

pickle.dump(tuple(cameras),fid)

fid.close()

#
# optionally go through and remove radial distortion from a set of images

import os

grab_dirs = [
    './david/grab_0/',
    './david/grab_1/',
    './david/grab_2/',
    './david/grab_3/',
    './david/grab_4/'
]

for src_dir in grab_dirs:
    dst_dir = src_dir.replace('./david/', './david_undistorted/')
    os.makedirs(dst_dir, exist_ok=True)
    
    imprefixL = 'frame_C0_'
    imprefixR = 'frame_C1_'

    images = glob.glob(src_dir + "*.png")
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        
        if imprefixL in fname:
            dst = cv2.undistort(img, Ks[0], dists[0], None, Ks[0])
        else:
            dst = cv2.undistort(img, Ks[1], dists[1], None, Ks[1])

        outname = os.path.join(dst_dir, os.path.basename(fname))
        cv2.imwrite(outname,dst)
        print(f"Saved {outname}")