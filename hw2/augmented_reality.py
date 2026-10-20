"""CMU 16-385: Computer Vision — Augmented Reality with Planar Homographies.

Python companion to Augmented Reality with Planar Homographies.ipynb.
Run this script to execute the notebook exercises in their original order.
Keep the complete repository together so bundled inputs remain available.
"""

# %% Local setup
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT / "hw2")
Path("results").mkdir(exist_ok=True)
print("Working directory:", Path.cwd())

# %% Notebook cell 5
# All assignment assets are already included in this repository.
DATA_DIR = "data"
RES_DIR = "results"
ROT_MATCHES_PATH = os.path.join(RES_DIR, "brief_rot_test.pkl")

# %% Notebook cell 7
import os
import numpy as np
import cv2
import skimage.color
import pickle
from matplotlib import pyplot as plt
import scipy
from skimage.util import montage
import time

PATCHWIDTH = 9

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def briefMatch(desc1,desc2,ratio):

    matches = skimage.feature.match_descriptors(desc1,desc2,
                                                'hamming',
                                                cross_check=True,
                                                max_ratio=ratio)
    return matches

def plotMatches(img1,img2,matches,locs1,locs2):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    skimage.feature.plot_matched_features(img1,img2,
                                          keypoints0=locs1,keypoints1=locs2,
                                          matches=matches,ax=ax,
                                          matches_color='r',only_matches=True)
    plt.show()
    return

def makeTestPattern(patchWidth, nbits):

    np.random.seed(0)
    compareX = patchWidth*patchWidth * np.random.random((nbits,1))
    compareX = np.floor(compareX).astype(int)
    np.random.seed(1)
    compareY = patchWidth*patchWidth * np.random.random((nbits,1))
    compareY = np.floor(compareY).astype(int)

    return (compareX, compareY)

def computePixel(img, idx1, idx2, width, center):

    # BRIEF indices contain one value; extract scalars for current NumPy.
    idx1 = int(np.asarray(idx1).item())
    idx2 = int(np.asarray(idx2).item())

    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0

def computeBrief(img, locs):

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])

    return desc, locs

def corner_detection(img, sigma):

    # fast method
    result_img = skimage.feature.corner_fast(img, n=PATCHWIDTH, threshold=sigma)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    return locs

# %% Notebook cell 17
def matchPics(I1, I2, ratio, sigma):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images (RGB or Grayscale uint8)
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    # ===== your code here! =====

    # convert to greyscale
    def to_grayscale(img):
      normalized = img.astype(np.float64) / 255.0
      # RGB has .ndim == 3
      return normalized if img.ndim != 3 else skimage.color.rgb2gray(normalized)

    grayscale1 = to_grayscale(I1)
    grayscale2 = to_grayscale(I2)

    # TODO: Detect features in both images
    locs1 = corner_detection(grayscale1, sigma)
    locs2 = corner_detection(grayscale2, sigma)

    # TODO: Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(grayscale1, locs1)
    desc2, locs2 = computeBrief(grayscale2, locs2)

    # TODO: Match features using the descriptors
    locs1 = np.asarray(locs1).reshape(-1, 2)
    locs2 = np.asarray(locs2).reshape(-1, 2)
    if len(desc1) == 0 or len(desc2) == 0:
      matches = np.empty((0, 2), dtype=int)
    else:
      matches = briefMatch(desc1, desc2, ratio)

    # ==== end of code ====

    return matches, locs1, locs2

# %% Notebook cell 19
def displayMatched(I1, I2, ratio, sigma):
    """
    Displays matches between two images

    Input
    -----
    I1, I2: Source images
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    """

    print('Displaying matches for ratio: ', ratio, ' and sigma: ', sigma)

    # ===== your code here! =====
    matches, locs1, locs2 = matchPics(I1, I2, ratio, sigma)
    plotMatches(I1, I2, matches, locs1, locs2)

    # ==== end of code ====

# %% Notebook cell 21
# Feel free to play around with these parameters
# BUT when submitting the report use the original images
image1_name = "cv_cover.jpg"
image2_name = "cv_desk.png"
ratio = 0.7
sigma = 0.15

image1_path = os.path.join(DATA_DIR, image1_name)
image2_path = os.path.join(DATA_DIR, image2_name)

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

#bgr to rgb
if len(image1.shape) == 3 and image1.shape[2] == 3:
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

if len(image2.shape) == 3 and image2.shape[2] == 3:
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

displayMatched(image1, image2, ratio, sigma)

# %% Notebook cell 23
image1_name = "cv_cover.jpg"
image2_name = "cv_desk.png"

image1_path = os.path.join(DATA_DIR, image1_name)
image2_path = os.path.join(DATA_DIR, image2_name)

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

#bgr to rgb
if len(image1.shape) == 3 and image1.shape[2] == 3:
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

if len(image2.shape) == 3 and image2.shape[2] == 3:
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# ===== your code here! =====
for sigma in [0.05, 0.15, 0.35]:
  print("\n")
  displayMatched(image1, image2, ratio=0.7, sigma=sigma)

print("\n\n\n")

for ratio in [0.5, 0.7, 0.9]:
  print("\n")
  displayMatched(image1, image2, ratio=ratio, sigma=0.15)

# ==== end of code ====

# %% Notebook cell 26
def briefRot(min_deg, max_deg, deg_inc, ratio, sigma, filename):
    """
    Tests Brief with rotations.

    Input
    -----
    min_deg: minimum degree to rotate image
    max_deg: maximum degree to rotate image
    deg_inc: number of degrees to increment when iterating
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    filename: filename of image to rotate

    """

    if not os.path.exists(RES_DIR):
      raise RuntimeError('RES_DIR does not exist. did you run all cells?')

    # Read the image and convert bgr to rgb
    image_path = os.path.join(DATA_DIR, filename)
    image = cv2.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    match_degrees = [] # stores the degrees of rotation
    match_counts = [] # stores the number of matches at each degree of rotation

    for i in range(min_deg, max_deg, deg_inc):
        print(i)

        # ===== your code here! =====
        # Rotate Image
        rotated = scipy.ndimage.rotate(image, i, reshape=True)

        # Match features in images
        matches, locs1, locs2 = matchPics(image, rotated, ratio, sigma)

        # Visualizes matches at at least 3 different orientations
        # to include in your report
        if i in [0, 30, 90]:
          plotMatches(image, rotated, matches, locs1, locs2)

        # Update match_degrees and match_counts
        match_degrees.append(i)
        match_counts.append(len(matches))

        # ==== end of code ====

    # Save to pickle file
    matches_to_save = [match_counts, match_degrees, deg_inc]
    write_pickle(ROT_MATCHES_PATH, matches_to_save)

def dispBriefRotHist(matches_path=ROT_MATCHES_PATH):
    # Check if pickle file exists
    if not os.path.exists(matches_path):
      raise RuntimeError('matches_path does not exist. did you call briefRot?')

    # Read from pickle file
    match_counts, match_degrees, deg_inc = read_pickle(matches_path)

    # Display histogram
    # Bins are centered and separated every 10 degrees
    plt.figure()
    bins = [x - deg_inc/2 for x in match_degrees]
    bins.append(bins[-1] + deg_inc)
    plt.hist(match_degrees, bins=bins, weights=match_counts, log=True)
    #plt.hist(match_degrees, bins=[10 * (x-0.5) for x in range(37)], weights=match_counts, log=True)
    plt.title("Histogram of BREIF matches")
    plt.ylabel("# of matches")
    plt.xlabel("Rotation (deg)")
    plt.tight_layout()

    output_path = os.path.join(RES_DIR, 'histogram.png')
    plt.savefig(output_path)

# %% Notebook cell 28
# defaults are:
# min_deg = 0
# max_deg = 360
# deg_inc = 10
# ratio = 0.7
# sigma = 0.15
# filename = 'cv_cover.jpg'

# Controls the rotation degrees
min_deg = 0
max_deg = 360
deg_inc = 10

# Brief feature descriptor and Fast feature detector paremeters
# (change these if you want to use different values)
ratio = 0.7
sigma = 0.15

# image to rotate and match
# (no need to change this but can if you want to experiment)
filename = 'cv_cover.jpg'

# Call briefRot
briefRot(min_deg, max_deg, deg_inc, ratio, sigma, filename)

# %% Notebook cell 30
dispBriefRotHist()

# %% Notebook cell 34
def computeH(x1, x2):
    """
    Compute the homography between two sets of points

    Input
    -----
    x1, x2: Sets of points

    Returns
    -------
    H2to1: 3x3 homography matrix that best transforms x2 to x1
    """

    if x1.shape != x2.shape:
      raise RuntimeError('number of points do not match')

    # ===== your code here! =====
    N = x1.shape[0]
    if N < 4: raise RuntimeError("at least four pairs are needed")

    A = np.zeros((2*N, 9), dtype=np.float64)
    for i in range(N):
      x, y = x2[i]
      u, v = x1[i]
      A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
      A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    _, _, VT = np.linalg.svd(A)
    H2to1 = VT[-1].reshape(3, 3)

    # ==== end of code ====

    return H2to1

# %% Notebook cell 36
def computeH_norm(x1, x2):
    """
    Compute the homography between two sets of points using normalization

    Input
    -----
    x1, x2: Sets of points

    Returns
    -------
    H2to1: 3x3 homography matrix that best transforms x2 to x1
    """

    # ===== your code here! =====

    # Compute the centroid of the points
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    shifted1 = x1 - centroid1
    shifted2 = x2 - centroid2

    # Normalize the points so that the largest distance from the
    # origin is equal to sqrt(2)
    largest_distance1 = np.max(np.linalg.norm(shifted1, axis=1))
    largest_distance2 = np.max(np.linalg.norm(shifted2, axis=1))
    if largest_distance1 == 0 or largest_distance2 == 0: raise RuntimeError("Division by 0")
    scale1, scale2 = np.sqrt(2)/largest_distance1, np.sqrt(2)/largest_distance2
    normalized1 = shifted1 * scale1
    normalized2 = shifted2 * scale2

    # Similarity transform 1
    T1 = np.array([[scale1, 0, -scale1 * centroid1[0]],
            [0, scale1, -scale1 * centroid1[1]],
            [0, 0, 1]])

    # Similarity transform 2
    T2 = np.array([[scale2, 0, -scale2 * centroid2[0]],
            [0, scale2, -scale2 * centroid2[1]],
            [0, 0, 1]])

    # Compute homography
    H = computeH(normalized1, normalized2)

    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H @ T2

    # ==== end of code ====

    return H2to1

# %% Notebook cell 38
def computeH_ransac(locs1, locs2, max_iters, inlier_tol):
  """
  Estimate the homography between two sets of points using ransac

  Input
  -----
  locs1, locs2: Lists of points
  max_iters: the number of iterations to run RANSAC for
  inlier_tol: the tolerance value for considering a point to be an inlier

  Returns
  -------
  bestH2to1: 3x3 homography matrix that best transforms locs2 to locs1
  inliers: indices of RANSAC inliers

  """

  # ===== your code here! =====

  # Compute the best fitting homography using RANSAC
  # given a list of matching points locs1 and loc2
  locs1 = np.asarray(locs1, dtype=np.float64)
  locs2 = np.asarray(locs2, dtype=np.float64)

  if locs1.shape != locs2.shape:
    raise RuntimeError('number of points do not match')

  N = locs1.shape[0]
  if N < 4: raise RuntimeError("at least four pairs are needed")

  points2 = np.hstack((locs2, np.ones((N, 1))))
  bestH2to1 = None
  best_inliers = None
  best_count = 0

  def find_inliers(H):
    proj = (H @ points2.T).T

    # Avoid division by zero
    valid = np.abs(proj[:, 2]) > 1e-10
    errors = np.full(N, np.inf)

    predicted = proj[valid, :2] / proj[valid, 2:3]
    errors[valid] = np.linalg.norm(predicted - locs1[valid], axis=1)
    return errors < inlier_tol

  for _ in range(max_iters):
    indices = np.random.choice(N, size=4, replace=False)
    sample1 = locs1[indices]
    sample2 = locs2[indices]

    valid_sample = True
    for sample in (sample1, sample2):
      for j in range(4):
        three = np.delete(sample, j, axis=0)
        matrix = np.hstack((three, np.ones((3, 1))))
        if np.linalg.matrix_rank(matrix) < 3: valid_sample = False

    if not valid_sample:
      continue

    try:
      H = computeH_norm(sample1, sample2)
    except (np.linalg.LinAlgError, RuntimeError):
      continue

    if not np.all(np.isfinite(H)):
      continue

    inliers = find_inliers(H)
    count = np.sum(inliers)

    if count > best_count:
      best_count = count
      bestH2to1 = H
      best_inliers = inliers.astype(int)

  if best_count < 4: raise RuntimeError("unable to find a valid homography")

  H = computeH_norm(locs1[best_inliers.astype(bool)], locs2[best_inliers.astype(bool)])
  inliers = find_inliers(H)

  if np.sum(inliers) >= best_count:
    bestH2to1 = H
    best_inliers = inliers.astype(int)

  # ==== end of code ====

  return bestH2to1, best_inliers

# %% Notebook cell 40
def compositeH(H2to1, template, img):
    """
    Returns the composite image.

    Input
    -----
    H2to1: Homography from image to template
    template: template image to be warped
    img: background image

    Returns
    -------
    composite_img: Composite image

    """

    # ===== your code here! =====
    # Create a composite image after warping the template image on top
    # of the image using the homography
    H = np.linalg.inv(H2to1)
    h, w = img.shape[:2]

    warped = cv2.warpPerspective(template, H, (w, h))

    mask = np.ones(template.shape[:2], dtype=np.uint8)
    mask = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST).astype(bool)

    composite_img = img.copy()
    composite_img[mask] = warped[mask]

    # ==== end of code ====

    return composite_img

# %% Notebook cell 42
def warpImage(ratio, sigma, max_iters, inlier_tol):
    """
    Warps hp_cover.jpg onto the book cover in cv_desk.png.

    Input
    -----
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    max_iters: the number of iterations to run RANSAC for
    inlier_tol: the tolerance value for considering a point to be an inlier

    """

    hp_cover = skimage.io.imread(os.path.join(DATA_DIR, 'hp_cover.jpg'))
    cv_cover = skimage.io.imread(os.path.join(DATA_DIR, 'cv_cover.jpg'))
    cv_desk = skimage.io.imread(os.path.join(DATA_DIR, 'cv_desk.png'))
    cv_desk = cv_desk[:, :, :3]

    # ===== your code here! =====

    # Match features between cv_desk and cv_cover using matchPics\
    matches, locs1, locs2 = matchPics(cv_desk, cv_cover, ratio, sigma)
    desk_points = locs1[matches[:, 0]][:, ::-1].astype(np.float64)
    cover_points = locs2[matches[:, 1]][:, ::-1].astype(np.float64)

    # Scale matched pixels in cv_cover to size of hp_cover
    h, w = cv_cover.shape[:2]
    hp_h, hp_w = hp_cover.shape[:2]
    cover_points[:, 0] *= hp_w / w
    cover_points[:, 1] *= hp_h / h

    # Get homography by RANSAC using computeH_ransac
    H, _ = computeH_ransac(cover_points, desk_points, max_iters, inlier_tol)

    # Overlay using compositeH to return composite_img
    composite_img = compositeH(H, hp_cover, cv_desk)

    # ==== end of code ====

    plt.imshow(composite_img)
    plt.show()

# %% Notebook cell 44
# defaults are:
# ratio = 0.7
# sigma = 0.15
# max_iters = 600
# inlier_tol = 1.0

# (no need to change this but can if you want to experiment)
ratio = 0.7
sigma = 0.15
max_iters = 600
inlier_tol = 1.0

warpImage(ratio, sigma, max_iters, inlier_tol)

# %% Notebook cell 46
# ===== your code here! =====
# Experiment with different max_iters and inlier_tol values.
# Include the result images in the write-up.
ratio = 0.7
sigma = 0.15

for max_iters in [1, 5, 20, 600]:
    print("\n")
    np.random.seed(0)
    print(f"max_iters = {max_iters}")
    plt.figure()
    warpImage(ratio, sigma, max_iters, inlier_tol=1.0)

print("\n\n\n")

for inlier_tol in [0.3, 1.0, 50.0, 100.0]:
    print("\n")
    np.random.seed(0)
    print(f"inlier_tol = {inlier_tol}")
    plt.figure()
    warpImage(ratio, sigma, max_iters=600, inlier_tol=inlier_tol)

# ==== end of code ====

# %% Notebook cell 51
def createPanorama(left_im, right_im, ratio, sigma, max_iters, inlier_tol):
    """
    Create a panorama augmented reality application by computing a homography
    and stitching together a left and right image.

    Input
    -----
    left_im: left image
    right_im: right image
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    max_iters: the number of iterations to run RANSAC for
    inlier_tol: the tolerance value for considering a point to be an inlier

    Returns
    -------
    panorama_im: Stitched together panorama

    """

    # ===== your code here! =====
    # match features between images
    matches, locs1, locs2 = matchPics(left_im, right_im, ratio, sigma)
    points1 = locs1[matches[:, 0]][:, ::-1]
    points2 = locs2[matches[:, 1]][:, ::-1]

    # Get homography by RANSAC using computeH_ransac
    H, _ = computeH_ransac(points1, points2, max_iters, inlier_tol)
    h1, w1 = left_im.shape[:2]
    h2, w2 = right_im.shape[:2]
    corners1 = np.array([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]], dtype=np.float64)
    corners2 = np.array([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]], dtype=np.float64)

    warped_corners = cv2.perspectiveTransform(corners2.reshape(-1, 1, 2), H).reshape(-1, 2)
    corners = np.vstack((corners1, warped_corners))
    x_min, y_min = np.floor(corners.min(axis=0)).astype(int)    # so buggy...
    x_max, y_max = np.ceil(corners.max(axis=0)).astype(int)

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    panorama_im = cv2.warpPerspective(right_im, T @ H,(int(x_max - x_min + 1), int(y_max - y_min + 1)))

    x, y = -x_min, -y_min
    panorama_im[y:y+h1, x:x+w1] = left_im

    # ==== end of code ====

    return panorama_im.astype(np.uint8)

# %% Notebook cell 53
left_im_path = os.path.join(DATA_DIR, 'chicago_left.jpg')
left_im = skimage.io.imread(left_im_path)
right_im_path = os.path.join(DATA_DIR, 'chicago_right.jpg')
right_im = skimage.io.imread(right_im_path)

# Feel free to adjust as needed
ratio = 0.7
sigma = 0.15
max_iters = 600
inlier_tol = 1.0

panorama_im = createPanorama(left_im, right_im, ratio, sigma, max_iters, inlier_tol)

plt.imshow(panorama_im)
plt.axis('off')
plt.show()

# %% Display any remaining figures outside Jupyter
plt.show()
