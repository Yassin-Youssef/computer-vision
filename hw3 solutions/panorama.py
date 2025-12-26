import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_h(img)#check here again 
    dy = filters.sobel_v(img)

    ### YOUR CODE HERE
    #compute 2x2 structure tensor entries
    Ixx = convolve(dx * dx, window, mode='reflect')
    Iyy = convolve(dy * dy, window, mode='reflect')
    Ixy = convolve(dx * dy, window, mode='reflect')
    #determinant and trace of structure tensor M
    detM = Ixx * Iyy - (Ixy ** 2)
    traceM = Ixx + Iyy
    response = detM - k * (traceM ** 2)#harris response
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []
    ### YOUR CODE HERE
    patch = patch.astype(np.float32)#converting to float
    mean = np.mean(patch)#computing the mean
    std = np.std(patch)
    if std == 0:#avoiding the divison to become 0
        std = 1.0
    normalized = (patch - mean) / std
    feature = normalized.flatten()
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Return descriptors in the same order/length as keypoints.
    Uses reflect padding so border keypoints are kept (no skipping).
    """
    image = image.astype(np.float32)
    half = patch_size // 2

    # reflect pad so a full patch is always available
    pad_img = np.pad(image, ((half, half), (half, half)), mode='reflect')

    desc = []
    for (y, x) in keypoints:          # (row=y, col=x)
        # window centered at (y,x) in the original image
        patch = pad_img[y : y + patch_size, x : x + patch_size]
        desc.append(desc_func(patch))

    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    if N == 0 or desc2.shape[0] == 0:#if they are empty then there are no matches
        return np.zeros((0, 2), dtype=int)
    #sorting the ditances
    order = np.argsort(dists, axis=1)
    nn1 = order[:, 0]
    nn2 = order[:, 1] if dists.shape[1] > 1 else order[:, 0]
    d1 = dists[np.arange(N), nn1]#best distance
    d2 = dists[np.arange(N), nn2] + 1e-12 # to second best distance
    ratio = d1 / d2 #ratio test
    keep = ratio < threshold
    matches = np.stack((np.where(keep)[0], nn1[keep]), axis=1).astype(int)
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)

    Return:
        H: a matrix of shape (P, P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    H, _, _, _ = np.linalg.lstsq(p2, p1, rcond=None)#solve for H in least squares
    H[2, 2] = 1.0#numerical stability
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    print(N)
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    ### YOUR CODE HERE
    rng = np.random.default_rng()                  # random number generator for RANSAC
    H = np.eye(3)                                  # best affine transform found so far
    best_inliers = np.zeros(N, dtype=bool)         # boolean mask for best inliers
    best_count = 0                                 # number of best inliers found

    for _ in range(n_iters):
        idx = rng.choice(N, size=n_samples, replace=False)  # randomly sample matches
        A = matched2[idx]                          # points from image 2
        B = matched1[idx]                          # corresponding points from image 1
        try:
            H_cand, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # estimate affine transform
        except np.linalg.LinAlgError:
            continue
        pred = (matched2 @ H_cand)[:, :2]          # transform all points using H_cand
        gt = matched1[:, :2]                       # ground truth points
        err = np.linalg.norm(pred - gt, axis=1)    # reprojection error
        inliers = err < threshold                  # determine inliers
        count = np.count_nonzero(inliers)          # count inliers
        if count > best_count:                     # update best model
            best_count = count
            best_inliers = inliers
            H = H_cand

    if best_count >= 3:
        A = matched2[best_inliers]                 # inlier points from image 2
        B = matched1[best_inliers]                 # inlier points from image 1
        H, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # re-fit using all inliers
    max_inliers = best_inliers
### END YOUR CODE

    ### END YOUR CODE
    print(H)
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    for r in range(rows):
        for c in range(cols):
            g_cell = G_cells[r, c]
            th_cell = theta_cells[r, c]
            bin_idx = (th_cell // degrees_per_bin).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            hist = np.bincount(bin_idx.ravel(),
                               weights=g_cell.ravel(),
                               minlength=n_bins)
            cells[r, c, :] = hist
    vec = cells.ravel().astype(float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1.0
    block = vec / norm
    ### YOUR CODE HERE

    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    # CHANGE: respect cval=-1 from warp_image (holes are -1, not 0)
    img1_mask = (img1_warped != -1)
    img2_mask = (img2_warped != -1)
    # Mask == 1 inside the image
    # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    left = int(left_margin)
    right = int(right_margin)
    left = max(0, min(left, out_W))
    right = max(0, min(right, out_W))
    w1_x = np.zeros(out_W, dtype=float)
    w2_x = np.zeros(out_W, dtype=float)
    w1_x[:left] = 1.0
    if right > left:
        L = right - left
        w1_x[left:right] = np.linspace(1.0, 0.0, L, endpoint=False)
    w1_x[right:] = 0.0
    w2_x = 1.0 - w1_x
    w1 = np.tile(w1_x, (out_H, 1)) * img1_mask
    w2 = np.tile(w2_x, (out_H, 1)) * img2_mask
    denom = w1 + w2
    denom[denom == 0] = 1.0
    merged = (img1_warped * w1 + img2_warped * w2) / denom
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    panorama = imgs[0]
    for i in range(len(imgs) - 1):
        m = matches[i]
        if m.size == 0:
            continue
        p1 = keypoints[i][m[:, 0]]
        p2 = keypoints[i+1][m[:, 1]]
        H, _ = ransac(p1, p2, m, n_iters=300, threshold=3.0)
        out_shape, offset = get_output_space(panorama, [imgs[i+1]], [H])
        pano_warped = warp_image(panorama, np.eye(3), out_shape, offset)
        next_warped = warp_image(imgs[i+1], H, out_shape, offset)
        panorama = linear_blend(pano_warped, next_warped)
    ### END YOUR CODE

    return panorama
