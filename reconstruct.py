import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

from camutils import triangulate

def decode(imprefix,start,threshold):
   """
   Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
   decode the binary sequence into a decimal value in (0,1023) for each pixel.
   Mark those pixels whose code is likely to be incorrect based on the user 
   provided threshold.  Images are assumed to be named "imageprefixN.png" where
   N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")

   Parameters
   ----------
   imprefix : str
      Image name prefix
   
   start : int
      Starting index
      
   threshold : float
      Threshold to determine if a bit is decodeable
      
   Returns
   -------
   code : 2D numpy.array (dtype=float)
      Array the same size as input images with entries in (0..1023)
      
   mask : 2D numpy.array (dtype=logical)
      Array indicating which pixels were correctly decoded based on the threshold
   
   """
   
   def graycode(gray_bits): # (10, 800, 1280)
      binary = np.zeros_like(gray_bits)
      binary[0] = gray_bits[0]
      for i in range(1, 10):
         binary[i, :] = np.logical_xor(binary[i - 1, :], gray_bits[i, :])
      
      return binary
   
   # we will assume a 10 bit code
   nbits = 10
   gray_bits = [] # (10, 800, 1280), 10 images of 800 x 1280
   mask = None # (800, 1280)
    
   for i in range(0, nbits):
         idx1 = start + 2 * i
         idx2 = start + 2 * i + 1
         img1 = plt.imread(f"{imprefix}{idx1:02d}_u.png").astype(float)
         img2 = plt.imread(f"{imprefix}{idx2:02d}_u.png").astype(float)
         
         # don't forget to convert images to grayscale / float after loading them in
         if img1.ndim == 3:
            img1 = img1[:, :, 0]
            img2 = img2[:, :, 0]
         
         diff = img1 - img2
         gray_img = diff > 0
         gray_bits.append(gray_img)
         greater_threshold = np.abs(diff) >= threshold
         
         if mask is None:
            mask = greater_threshold
         else:
            mask = np.logical_and(mask, greater_threshold)

   gray_bits = np.array(gray_bits)
   binary_bits = graycode(gray_bits) # (10, 800, 1280)
   
   code = np.zeros(binary_bits.shape[1:], dtype=np.float32)

   for i in range(10):
      code += binary_bits[i, :, :] * (2 ** (9 - i))
        
   return code,mask

def reconstruct(image_dir, imprefixL,imprefixR,decode_threshold,color_threshold,camL,camR):
    """
    Performing matching and triangulation of points on the surface using structured
    illumination. This function decodes the binary graycode patterns, matches 
    pixels with corresponding codes, and triangulates the result.
    
    The returned arrays include 2D and 3D coordinates of only those pixels which
    were triangulated where pts3[:,i] is the 3D coordinte produced by triangulating
    pts2L[:,i] and pts2R[:,i]

    Parameters
    ----------
    imprefixL, imprefixR : str
        Image prefixes for the coded images from the left and right camera
        
    threshold : float
        Threshold to determine if a bit is decodeable
   
    camL,camR : Camera
        Calibration info for the left and right cameras
        
    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)
        The 2D pixel coordinates of the matched pixels in the left and right
        image stored in arrays of shape 2xN
        
    pts3 : 2D numpy.array (dtype=float)
        Triangulated 3D coordinates stored in an array of shape 3xN
        
    """

    # Decode the H and V coordinates for the two views
    HL,HLmask = decode(imprefixL,0,decode_threshold)
    VL,VLmask = decode(imprefixL,20,decode_threshold)
    
    HR,HRmask = decode(imprefixR,0,decode_threshold)
    VR,VRmask = decode(imprefixR,20,decode_threshold)
    
    colorL_obj = plt.imread(image_dir + "color_C0_01_u.png")[:, :, :3]
    # Temporarily use grab_0's background image
    colorL_bg = plt.imread('./david_undistorted/grab_0/color_C0_00_u.png')[:, :, :3]
   
   #  colorL_bg  = plt.imread(image_dir + "color_C0_00_u.png")[:, :, :3]

    colorR_obj = plt.imread(image_dir + "color_C1_01_u.png")[:, :, :3]
   #  colorR_bg  = plt.imread(image_dir + "color_C1_00_u.png")[:, :, :3]

    colorR_bg = plt.imread('./david_undistorted/grab_0/color_C1_00_u.png')[:, :, :3]

    # Construct the combined 20 bit code C = 1024*H + V and mask for each view
    
    # Foreground masks
    diffL = np.linalg.norm(colorL_obj - colorL_bg, axis=2)
    diffR = np.linalg.norm(colorR_obj - colorR_bg, axis=2)

    fgMaskL = diffL > color_threshold
    fgMaskR = diffR > color_threshold

    CL = HL * 1024 + VL
    CR = HR * 1024 + VR
    
    maskL = HLmask & VLmask & fgMaskL
    maskR = HRmask & VRmask & fgMaskR
    
    # Find the indices of pixels in the left and right code image that 
    # have matching codes. If there are multiple matches, just
    # choose one arbitrarily.
    h, w = CL.shape
    
    
    CLf = CL.reshape(-1)
    CRf = CR.reshape(-1)

    maskLf = maskL.reshape(-1)
    maskRf = maskR.reshape(-1)

    # Only keep valid pixels
    validL = np.where(maskLf)[0]
    validR = np.where(maskRf)[0]

    codesL = CLf[validL]
    codesR = CRf[validR]

    _, commL, commR = np.intersect1d(
        codesL,
        codesR,
        return_indices=True
    )
    
    matchL = validL[commL]
    matchR = validR[commR]
    
    # Let CL and CR be the flattened arrays of codes for the left and right view
    # Suppose you have computed arrays of indices matchL and matchR so that 
    # CL[matchL[i]] == CR[matchR[i]] for all i.  The code below gives one approach
    # to generating the corresponding pixel coordinates for the matched pixels.
    
    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))
    pts2R = np.concatenate((xx[matchR],yy[matchR]),axis=1).T
    pts2L = np.concatenate((xx[matchL],yy[matchL]),axis=1).T
    
    # Save colors
    
    color_flat = colorL_obj.reshape(-1, 3)
    colors = color_flat[matchL].T
 
    # Now triangulate the points
    pts3 = triangulate(pts2L, camL, pts2R, camR)
    
    return pts2L,pts2R,pts3,colors