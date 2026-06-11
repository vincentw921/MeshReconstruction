import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from scipy.spatial import cKDTree

class InsufficientCorrespondencesError(ValueError):
    """Raised when too few clicked pairs survive distance filtering."""


def click_correspondences(imgA, imgB):
    """
    Click alternating correspondences:
        A1, B1, A2, B2, ...

    Press Done when finished.

    Returns:
        ptsA: 2 x N
        ptsB: 2 x N
    """

    ptsA = []
    ptsB = []
    artists = []
    click_count = 0
    finished = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.18)

    axA, axB = axes
    axA.imshow(imgA)
    axB.imshow(imgB)

    for ax in axes:
        ax.axis("off")

    def update_titles():
        k = min(len(ptsA), len(ptsB)) + 1

        if click_count % 2 == 0:
            axA.set_title(f"Scan A: click point {k}")
            axB.set_title("Scan B: waiting")
        else:
            axA.set_title("Scan A: waiting")
            axB.set_title(f"Scan B: click matching point {k}")

        fig.canvas.draw_idle()

    def draw_point(ax, x, y, label):
        scat = ax.scatter(x, y, c="red", s=5)
        text = ax.text(x, y, str(label), color="yellow", fontsize=5)
        artists.append((scat, text))

    def onclick(event):
        nonlocal click_count
        
        # Skip if non-click event or finished
        if finished or event.xdata is None or event.ydata is None:
            return

        # Ignore button clicks
        if event.inaxes not in [axA, axB]:
            return

        # Check for clicks on images
        if click_count % 2 == 0:
            if event.inaxes != axA:
                return

            ptsA.append([event.xdata, event.ydata])
            draw_point(axA, event.xdata, event.ydata, len(ptsA))
            click_count += 1

        else:
            if event.inaxes != axB:
                return

            ptsB.append([event.xdata, event.ydata])
            draw_point(axB, event.xdata, event.ydata, len(ptsB))
            click_count += 1

        update_titles()

    def undo(event):
        nonlocal click_count

        if click_count == 0:
            return

        # Remove last visual marker
        scat, text = artists.pop()
        scat.remove()
        text.remove()

        # Remove last stored point
        if click_count % 2 == 1:
            # Last click was on A
            ptsA.pop()
        else:
            # Last click was on B
            ptsB.pop()

        click_count -= 1
        update_titles()

    def done(event):
        nonlocal finished

        if len(ptsA) != len(ptsB):
            print("You have an unmatched point. Click the matching point or undo.")
            return

        if len(ptsA) < 3:
            print("Need at least 3 correspondences for SVD alignment.")
            return

        finished = True
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)
        
        
    # Create undo and finish buttons

    ax_undo = plt.axes([0.35, 0.04, 0.12, 0.07])
    ax_done = plt.axes([0.53, 0.04, 0.12, 0.07])

    btn_undo = Button(ax_undo, "Undo")
    btn_done = Button(ax_done, "Done")

    btn_undo.on_clicked(undo)
    btn_done.on_clicked(done)

    update_titles()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()

    while not finished and plt.fignum_exists(fig.number):
        plt.pause(0.1)

    if len(ptsA) != len(ptsB) or len(ptsA) < 3:
        raise ValueError("Correspondence selection was cancelled or incomplete.")

    return np.asarray(ptsA, dtype=float).T, np.asarray(ptsB, dtype=float).T


def _nearest_reconstructed_points(clicked_pts2, pts2L, pts3):
    """Return the nearest reconstructed 3D point and pixel distance per click."""
    clicked_pts2 = np.asarray(clicked_pts2, dtype=float)
    pts2L = np.asarray(pts2L, dtype=float)
    pts3 = np.asarray(pts3, dtype=float)

    selected_pts3 = np.empty((3, clicked_pts2.shape[1]), dtype=pts3.dtype)
    distances = np.empty(clicked_pts2.shape[1], dtype=float)

    for i in range(clicked_pts2.shape[1]):
        offset = pts2L - clicked_pts2[:, i:i + 1]
        squared_distances = np.sum(offset ** 2, axis=0)
        idx = np.argmin(squared_distances)

        selected_pts3[:, i] = pts3[:, idx]
        distances[i] = np.sqrt(squared_distances[idx])

    return selected_pts3, distances

def click_pairs_to_3d(
    clicked_ptsA,
    pts2A,
    pts3A,
    clicked_ptsB,
    pts2B,
    pts3B,
    max_dist=5
):
    """
    Convert paired clicks to 3D and reject a pair if either click is too far.

    Returns two aligned 3 x M arrays suitable for rigid_alignment/SVD.
    """
    if clicked_ptsA.shape != clicked_ptsB.shape:
        raise ValueError("The two scans must have the same number of clicks.")

    selected_A, distances_A = _nearest_reconstructed_points(
        clicked_ptsA, pts2A, pts3A
    )
    selected_B, distances_B = _nearest_reconstructed_points(
        clicked_ptsB, pts2B, pts3B
    )

    keep = (distances_A <= max_dist) & (distances_B <= max_dist)

    kept_count = np.count_nonzero(keep)
    if kept_count < 3:
        raise InsufficientCorrespondencesError(
            "Need at least 3 valid click pairs for SVD alignment; "
            f"only {kept_count} passed max_dist={max_dist}."
        )

    return selected_A[:, keep], selected_B[:, keep]


def select_3d_correspondences(
    imgA,
    pts2A,
    pts3A,
    imgB,
    pts2B,
    pts3B,
    max_dist=5
):
    """
    Collect valid 3D correspondence pairs, restarting after a bad selection.
    """
    while True:
        clicked_ptsA, clicked_ptsB = click_correspondences(imgA, imgB)

        try:
            return click_pairs_to_3d(
                clicked_ptsA,
                pts2A,
                pts3A,
                clicked_ptsB,
                pts2B,
                pts3B,
                max_dist=max_dist,
            )
        except InsufficientCorrespondencesError as error:
            print(f"{error} Restarting correspondence selection.")


def svd_alignment(P, Q):
    """
    Find R, t such that Q ≈ R @ P + t.

    P: 3 x N points from scan A
    Q: 3 x N corresponding points from scan B
    """
    assert P.shape == Q.shape
    assert P.shape[0] == 3
    
    # Compute Centroids

    cP = np.mean(P, axis=1, keepdims=True)
    cQ = np.mean(Q, axis=1, keepdims=True)

    # Translate Input Points to Centroids

    P0 = P - cP
    Q0 = Q - cQ
    
    # Compute Covariance Matrix

    H = P0 @ Q0.T
    
    # Compute SVD using Numpy

    U, S, Vt = np.linalg.svd(H)
    
    # Extract Optimal Rotation

    R = Vt.T @ U.T

    # Fix reflection if needed
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    # Extract Translation

    t = cQ - R @ cP

    return R, t

def transform_points(pts3, R, t):
    """
    Apply transformation to 3D points.
    """
    return R @ pts3 + t

def icp_point_to_point(
    source_pts,
    target_pts,
    R_init,
    t_init,
    max_iters=30,
    max_match_dist=1.5
):
    """
    Refine alignment from source_pts to target_pts.

    source_pts : 3 x N
    target_pts : 3 x M
    """

    R_total = R_init.copy()
    t_total = t_init.copy()
    
    # Transform points using the current R and t estimate.
    source_aligned = transform_points(
        source_pts,
        R_total,
        t_total
    )

    # Maintain a point tree to quickly find nearest points
    tree = cKDTree(target_pts.T)

    for it in range(max_iters):

        # Find nearest target point for every source point
        dists, nn_idx = tree.query(
            source_aligned.T,
            k=1
        )

        # Reject bad matches
        keep = dists < max_match_dist

        if np.sum(keep) < 6:
            print("ICP stopped: too few valid matches.")
            break

        P = source_aligned[:, keep]
        Q = target_pts[:, nn_idx[keep]]

        # Compute small alignment difference
        dR, dt = svd_alignment(P, Q)

        # Apply correction
        source_aligned = dR @ source_aligned + dt

        # Update total transform
        R_total = dR @ R_total
        t_total = dR @ t_total + dt

    return R_total, t_total, source_aligned