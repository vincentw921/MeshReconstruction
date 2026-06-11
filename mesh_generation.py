from reconstruct import reconstruct
import pickle

from scipy.spatial import Delaunay
import numpy as np

def get_corresponding_points(image_dir, decode_threshold = 0.03, color_threshold = 0.1):
    imprefixL = image_dir + 'frame_C0_'
    imprefixR = image_dir + 'frame_C1_'
    
    fid = open('/Users/vincentwang/Desktop/CS117/Final Project/calibration.pickle','rb')
    (camC0,camC1) = pickle.load(fid)
    fid.close()

    pts2L,pts2R,pts3,colors = reconstruct(image_dir, imprefixL,imprefixR,decode_threshold,color_threshold,camC0,camC1)

    # Get min and max for bounding box (X, Y, Z)
    X = pts3[0, :]
    Y = pts3[1, :]
    Z = pts3[2, :]
    bounding_box = [X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()]
    print(f"Largest Bounding Box: {bounding_box}")
    return pts3, pts2L, pts2R, colors

def bounding_box_pruning(pts3, pts2L, pts2R, colors, bounding_box=None):
    # Bounding box pruning

    X = pts3[0, :]
    Y = pts3[1, :]
    Z = pts3[2, :]

    if bounding_box is None:
        bounding_box = (X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max())
    
    keep = (
        (X >= bounding_box[0]) & (X <= bounding_box[1]) &
        (Y >= bounding_box[2]) & (Y <= bounding_box[3]) &
        (Z >= bounding_box[4]) & (Z <= bounding_box[5])
    )
    
    pts3_pruned = pts3[:, keep]
    pts2L_pruned = pts2L[:, keep]
    pts2R_pruned = pts2R[:, keep]
    colors_pruned = colors[:, keep]

    return pts3_pruned, pts2L_pruned, pts2R_pruned, colors_pruned

def create_mesh(pts3, pts2L, colors, trithresh: float = 100):
    tri = Delaunay(pts2L.T)
    triangles = tri.simplices
    
    good_triangles = []

    for t in triangles:
        p0 = pts3[:, t[0]]
        p1 = pts3[:, t[1]]
        p2 = pts3[:, t[2]]

        e01 = np.linalg.norm(p0 - p1)
        e12 = np.linalg.norm(p1 - p2)
        e20 = np.linalg.norm(p2 - p0)

        if e01 < trithresh and e12 < trithresh and e20 < trithresh:
            good_triangles.append(t)

    triangles = np.array(good_triangles, dtype=int)

    # remove any points which are not refenced in any triangle

    used = np.unique(triangles.reshape(-1))

    pts3_clean = pts3[:, used]
    pts2L_clean = pts2L[:, used]
    colors_clean = colors[:, used]

    old_to_new = np.zeros(pts3.shape[1], dtype=int)
    old_to_new[used] = np.arange(len(used))

    triangles_clean = old_to_new[triangles]
    
    return pts3_clean, triangles_clean, colors_clean, pts2L_clean


def mesh_generation(image_dir, result_file, bounding_box = None, threshold = 0.03, trithresh = 100):
    
    #
    # Reconstruct from the two views.
    #

    pts3, pts2L, pts2R, colors = get_corresponding_points(image_dir, threshold)

    pts3_pruned, pts2L_pruned, pts2R_pruned, colors_pruned = bounding_box_pruning(pts3, pts2L, pts2R, colors, bounding_box)

    pts3_clean, triangles_clean, colors_clean, pts2L_clean = create_mesh(pts3_pruned, pts2L_pruned, colors_pruned, trithresh)

    pickle.dump((pts3_clean, triangles_clean, colors_clean, pts2L_clean), open(result_file, 'wb'))
    
    return pts3_clean, colors_clean, triangles_clean

def mesh_smoothing(pts3, triangles, iterations = 1):
    smoothed_pts3 = pts3.copy()
    for _ in range(iterations):
        for j in range(len(triangles)):
            t = triangles[j]
            p0 = smoothed_pts3[:, t[0]]
            p1 = smoothed_pts3[:, t[1]]
            p2 = smoothed_pts3[:, t[2]]
            # Compute the centroid of the triangle
            centroid = (p0 + p1 + p2) / 3
            # Move the vertices towards the centroid
            smoothed_pts3[:, t[0]] = p0 + 0.1 * (centroid - p0)
            smoothed_pts3[:, t[1]] = p1 + 0.1 * (centroid - p1)
            smoothed_pts3[:, t[2]] = p2 + 0.1 * (centroid - p2)
    return smoothed_pts3