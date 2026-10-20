"""CMU 16-385: Computer Vision — 3D Reconstruction.

Python companion to 3D Reconstruction.ipynb.
Run this script to execute the notebook exercises in their original order.
Keep the complete repository together so bundled inputs remain available.
"""

# %% Local setup
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT / "hw3")
Path("results").mkdir(exist_ok=True)
print("Working directory:", Path.cwd())

# %% Notebook cell 7
import os
import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


connections_3d = [[0,1], [1,3], [2,3], [2,0], [4,5], [6,7], [8,9], [9,11], [10,11], [10,8], [0,4], [4,8],
                  [1,5], [5,9], [2,6], [6,10], [3,7], [7,11]]
color_links = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),(0,0,255),(255,0,255),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255)]
colors = ['blue','blue','blue','blue','red','magenta','green','green','green','green','red','red','red','red','magenta','magenta','magenta','magenta']


def visualize_keypoints(image, pts, Threshold=100):
    '''
    This function visualizes the 2d keypoint pairs in connections_3d
    (as define above) whose match score lies above a given Threshold
    in an OpenCV GUI frame, against an image background.

    :param image: image as a numpy array, of shape (height, width, 3) where 3 is the number of color channels
    :param pts: np.array of shape (num_points, 3)
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(12):
        cx, cy = pts[i][0:2]
        if pts[i][2]>Threshold:
            cv2.circle(image,(int(cx),int(cy)),5,(0,255,255),5)

    for i in range(len(connections_3d)):
        idx0, idx1 = connections_3d[i]
        if pts[idx0][2]>Threshold and pts[idx1][2]>Threshold:
            x0, y0 = pts[idx0][0:2]
            x1, y1 = pts[idx1][0:2]
            cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), color_links[i], 2)

    plt.imshow(image)
    plt.show()

    return image


def plot_3d_keypoint(pts_3d):
    '''
    this function visualizes 3d keypoints on a matplotlib 3d axes

    :param pts_3d: np.array of shape (num_points, 3)
    '''
    fig = plt.figure()
    num_points = pts_3d.shape[0]
    ax = fig.add_subplot(111, projection='3d')
    for j in range(len(connections_3d)):
        index0, index1 = connections_3d[j]
        xline = [pts_3d[index0,0], pts_3d[index1,0]]
        yline = [pts_3d[index0,1], pts_3d[index1,1]]
        zline = [pts_3d[index0,2], pts_3d[index1,2]]
        ax.plot(xline, yline, zline, color=colors[j])
    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def calc_epi_error(pts1_homo, pts2_homo, F):
    '''
    Helper function to calcualte the sum of squared distance between the
    corresponding points and the estimated epipolar lines.

    pts1_homo \dot F.T \dot pts2_homo = 0

    :param pts1_homo: of shape (num_points, 3); in homogeneous coordinates, not normalized.
    :param pts2_homo: same specification as to pts1_homo.
    :param F: Fundamental matrix
    '''

    line1s = pts1_homo.dot(F.T)
    dist1 = np.square(np.divide(np.sum(np.multiply(
        line1s, pts2_homo), axis=1), np.linalg.norm(line1s[:, :2], axis=1)))

    line2s = pts2_homo.dot(F)
    dist2 = np.square(np.divide(np.sum(np.multiply(
        line2s, pts1_homo), axis=1), np.linalg.norm(line2s[:, :2], axis=1)))

    ress = (dist1 + dist2).flatten()
    return ress


def toHomogenous(pts):
    """
    Adds a stack of ones at the end, to turn a set of points into a set of
    homogeneous points.

    :params pts: in shape (num_points, 2).
    """
    return np.vstack([pts[:,0],pts[:,1],np.ones(pts.shape[0])]).T.copy()


def _epipoles(E):
    """
    gets the epipoles from the Essential Matrix.

    :params E: Essential matrix.
    """
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]
    return e1, e2


def displayEpipolarF(I1, I2, F, points):
    """
    GUI interface you may use to help you verify your calculated fundamental
    matrix F. Select a point I1 in one view, and it should correctly correspond
    to the displayed point in the second view.
    """
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('The point you selected:')
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')

    plt.sca(ax1)

    colors = ['r','g','b','y','m','k']
    for i, out in enumerate(points):
      x, y = out #[0]

      xc = x
      yc = y
      v = np.array([xc, yc, 1])
      l = F.dot(v)
      s = np.sqrt(l[0]**2+l[1]**2)

      if s==0:
          print('Zero line vector in displayEpipolar')

      l = l/s

      if l[0] != 0:
          ye = sy-1
          ys = 0
          xe = -(l[1] * ye + l[2])/l[0]
          xs = -(l[1] * ys + l[2])/l[0]
      else:
          xe = sx-1
          xs = 0
          ye = -(l[0] * xe + l[2])/l[1]
          ys = -(l[0] * xs + l[2])/l[1]

      # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
      ax1.plot(x, y, '*', markersize=6, linewidth=2, color=colors[i%len(colors)])
      ax2.plot([xs, xe], [ys, ye], linewidth=2, color=colors[i%len(colors)])
    plt.draw()


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F

def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
    return r

def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000,
        disp=False
    )
    return _singularize(f.reshape([3, 3]))


# Used for Epipolar Correspondence
def epipolarMatchGUI(I1, I2, F, points, epipolarCorrespondence):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('The point you selected:')
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image \nand that the corresponding point matches')

    plt.sca(ax1)

    colors = ['r','g','b','y','m','k']

    for i, out in enumerate(points):
      x, y = out

      xc = int(x)
      yc = int(y)
      v = np.array([xc, yc, 1])
      l = F.dot(v)
      s = np.sqrt(l[0]**2+l[1]**2)

      if s==0:
          print('Zero line vector in displayEpipolar')

      l = l/s

      if l[0] != 0:
          ye = sy-1
          ys = 0
          xe = -(l[1] * ye + l[2])/l[0]
          xs = -(l[1] * ys + l[2])/l[0]
      else:
          xe = sx-1
          xs = 0
          ye = -(l[0] * xe + l[2])/l[1]
          ys = -(l[0] * xs + l[2])/l[1]

      ax1.plot(x, y, '*', markersize=6, linewidth=2, color=colors[i%len(colors)])
      ax2.plot([xs, xe], [ys, ye], linewidth=2, color=colors[i%len(colors)])

      # draw points
      x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
      ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
      plt.draw()

# %% Notebook cell 10
def eightpoint(pts1, pts2, M):
  '''
  Q1.1: Eight Point Algorithm
  Input:  pts1, Nx2 Matrix
          pts2, Nx2 Matrix
          M, a scalar parameter computed as max(imwidth, imheight)
  Output: F, the fundamental matrix

  HINTS:
  (1) Normalize the input pts1 and pts2 using the matrix T.
  (2) Setup the eight point algorithm's equation.
  (3) Solve for the least square solution using SVD.
  (4) Use the function `_singularize` (provided in the helper functions above) to enforce the singularity condition.
  (5) Use the function `refineF` (provided in the helper functions above) to refine the computed fundamental matrix.
      (Remember to use the normalized points instead of the original points)
  (6) Unscale the fundamental matrix by the lower right corner element
  '''

  # let's first define our T
  T = np.diag([1/M, 1/M, 1])

  pts1_norm = pts1 / M  # (N, 2)
  pts2_norm = pts2 / M  # (N, 2)

  x1, y1 = pts1_norm.T
  x2, y2 = pts2_norm.T

  A = np.vstack([x2 * x1, x2 * y1, x2,
          y2 * x1, y2 * y1, y2,
          x1, y1, np.ones(len(pts1))]).T

  _, _, Vt = np.linalg.svd(A)
  F = Vt[-1].reshape(3, 3)
  F = _singularize(F)
  F = refineF(F, pts1_norm, pts2_norm)
  F = T.T @ F @ T
  F = F / F[2, 2]   # we need F[2, 2] == 1

  return F

# %% Notebook cell 12
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')

F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
print(f'recovered F:\n{F.round(4)}')

# Simple Tests to verify your implementation:
pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

assert F.shape == (3, 3), "F is wrong shape"
assert F[2, 2] == 1, "F_33 != 1"
assert np.linalg.matrix_rank(F) == 2, "F should have rank 2"
assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1, "F error is too high to be accurate"

# %% Notebook cell 14
# the points in im1, whose correnponding epipolar line in im2 you'd like to verify
point = [(50,190),(200, 200), (400,180), (350,350), (520, 220)]
# feel free to change these point, to verify different point correspondences
displayEpipolarF(im1, im2, F, point)

# %% Notebook cell 17
def essentialMatrix(F, K1, K2):
  '''
  Q2.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
  '''
  E = K2.T @ F @ K1
  E = E / E[2, 2]

  return E

# %% Notebook cell 19
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')

F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
E = essentialMatrix(F, K1, K2)
print(f'recovered E:\n{E.round(4)}')

# Simple Tests to verify your implementation:
assert(E[2, 2] == 1)
assert(np.linalg.matrix_rank(E) == 2)

# %% Notebook cell 21
def triangulate(C1, pts1, C2, pts2):
  '''
  Q2.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
  Input:  C1, the 3x4 camera matrix
          pts1, the Nx2 matrix with the 2D image coordinates per row
          C2, the 3x4 camera matrix
          pts2, the Nx2 matrix with the 2D image coordinates per row
  Output: P, the Nx3 matrix with the corresponding 3D points per row
          err, the reprojection error.

  Hints:
  (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
  (2) Solve for the least square solution using np.linalg.svd
  (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from
      homogeneous coordinates to non-homogeneous ones)
  (4) Keep track of the 3D points and projection error, and continue to next point
  (5) You do not need to follow the exact procedure above.
  '''
  N = len(pts1)
  P = np.ones((N, 3))
  err = 0

  for i in range(N):
    x1, y1 = pts1[i]
    x2, y2 = pts2[i]

    # form A
    A = np.vstack([y1 * C1[2] - C1[1],
            C1[0] - x1 * C1[2],
            y2 * C2[2] - C2[1],
            C2[0] - x2 * C2[2]])
    _, _, Vt = np.linalg.svd(A)
    w = Vt[-1]
    w = w / w[-1]
    P[i] = w[:3]

    x1_h = C1 @ w
    x2_h = C2 @ w
    x1_h = x1_h[:2] / x1_h[-1]
    x2_h = x2_h[:2] / x2_h[-1]

    err += np.sum((pts1[i] - x1_h) ** 2)
    err += np.sum((pts2[i] - x2_h) ** 2)

  return P, err

# %% Notebook cell 23
def camera2(E):
  """helper function to find the 4 possibile M2 matrices"""
  U,S,V = np.linalg.svd(E)
  m = S[:2].mean()
  E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
  U,S,V = np.linalg.svd(E)
  W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

  if np.linalg.det(U.dot(W).dot(V))<0:
      W = -W

  M2s = np.zeros([3,4,4])
  M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  return M2s


def findM2(F, pts1, pts2, intrinsics):
    '''
    Q2.3: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of which has positive z values (image in front of both cameras).
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    '''

    K1, K2 = intrinsics['K1'], intrinsics['K2']

    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)

    M2 = None
    C2 = None
    P = None

    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = K1 @ M1

    best_err = float('inf')

    for i in range(4):
      M2_i = M2s[:, :, i]
      C2_i = K2 @ M2_i
      P_i, err = triangulate(C1, pts1, C2_i, pts2)

      P_h = np.hstack((P_i, np.ones((len(P_i), 1))))
      P_camera2 = (M2_i @ P_h.T).T

      if (np.all(P_i[:, 2] > 0) and np.all(P_camera2[:, 2] > 0) and err < best_err):
        best_err = err
        M2 = M2_i
        C2 = C2_i
        P = P_i

    print("best M2", M2)
    return M2, C2, P

# %% Notebook cell 25
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')

F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

M2, C2, P = findM2(F, pts1, pts2, intrinsics)

# Simple Tests to verify your implementation:
M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
C1 = K1.dot(M1)
C2 = K2.dot(M2)
P_test, err = triangulate(C1, pts1, C2, pts2)
assert(err < 500)

# %% Notebook cell 28
def epipolarCorrespondence(im1, im2, F, x1, y1):
  '''
  Q3.1: 3D visualization of the temple images.
  Input:  im1, the first image
          im2, the second image
          F, the fundamental matrix
          x1, x-coordinates of a pixel on im1
          y1, y-coordinates of a pixel on im1
  Output: x2, x-coordinates of the pixel on im2
          y2, y-coordinates of the pixel on im2

  Hints:
  (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
  (2) Search along this line to check nearby pixel intensity (you can define a search window) to
      find the best matches
  (3) Use gaussian weighting to weight the pixel simlairty

  '''
  x1, y1 = int(x1), int(y1)   # without this, next section gives overflow error...

  r = 7
  search_range = 60
  sigma = 4

  a, b, c = F @ np.array([x1, y1, 1])
  window1 = im1[y1 - r: y1 + r + 1, x1 - r: x1 + r + 1]
  x2 = None
  y2 = None

  g = cv2.getGaussianKernel(2 * r + 1, sigma)
  weights = (g @ g.T)[:, :, None]

  height, width = im2.shape[:2]
  best = float('inf')

  for i in range(-search_range, search_range + 1):
    if (abs(a) > abs(b)):
      y_i = y1 + i
      x_i = round(-(b * y_i + c) / a)
    else:
      x_i = x1 + i
      y_i = round(-(a * x_i + c) / b)

    # skip positions out of bounds!
    if (x_i < r or x_i >= width - r or y_i < r or y_i >= height - r): continue

    window2 = im2[y_i - r: y_i + r + 1, x_i - r: x_i + r + 1]
    curr = np.sum(weights * (window1 - window2)**2)
    if (curr < best):
      best = curr
      x2 = x_i
      y2 = y_i

  return x2, y2

# %% Notebook cell 30
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')

F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

print(F)

# Simple Tests to verify your implementation:
x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
print(x2, y2)
assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)

# %% Notebook cell 32
# the points in im1 whose correnponding epipolar line in im2 you'd like to verify
points = [(50,190), (200, 200), (400,180), (350,350), (520, 220)]
# feel free to change these points to verify different point correspondences
epipolarMatchGUI(im1, im2, F, points, epipolarCorrespondence)

# %% Notebook cell 34
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):
  '''
  Q3.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
  Input:  temple_pts1, chosen points from im1
          intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
          F, the fundamental matrix
          im1, the first image
          im2, the second image
  Output: P (Nx3) the recovered 3D points

  Hints:
  (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
  (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
      matrix and use triangulate to find the 3D points.
  (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
  (4) As a reference, our solution's best error is around ~750 on the 3D points.
  '''
  N = len(temple_pts1)
  temple_pts2 = np.ones((N, 2))

  for i in range(N):
    x1, y1 = temple_pts1[i]
    x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
    temple_pts2[i] = np.array([x2, y2])

  M2, C2, P = findM2(F, temple_pts1, temple_pts2, intrinsics)
  return P

# %% Notebook cell 36
temple_coords = np.load('data/templeCoords.npz') # Loading temple coordinates
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')

# ----- TODO -----
# Call eightpoint to get the F matrix
# Call compute3D_pts to get the 3D points and visualize using matplotlib scatter
# hint: you can change the viewpoint of a matplotlib 3d axes using
# `ax.view_init(azim, elev)` where azim is the rotation around the vertical z
# axis, and elev is the angle of elevation from the x-y plane

temple_pts1 = np.hstack([temple_coords['x1'], temple_coords['y1']])

M = np.max([*im1.shape, *im2.shape])
F = eightpoint(pts1, pts2, M)

P = compute3D_pts(temple_pts1, intrinsics, F, im1, im2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, c='c', depthshade=True)
plt.draw()

# also show a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, c='c', depthshade=True)
ax.view_init(30, 0)
plt.draw()

# %% Notebook cell 39
def ransacF(pts1, pts2, M, nIters=100, tol=10):
  '''
  Input:  pts1, Nx2 Matrix
          pts2, Nx2 Matrix
          M, a scaler parameter
          nIters, Number of iterations of the Ransac
          tol, tolerence for inliers
  Output: F, the fundamental matrix
          inliers, Nx1 bool vector set to true for inliers

  '''
  N = pts1.shape[0]
  pts1_homo, pts2_homo = toHomogenous(pts1), toHomogenous(pts2)
  best_inlier = 0
  inlier_curr = None

  for i in range(nIters):
      choice = np.random.choice(range(pts1.shape[0]), 8)
      pts1_choice = pts1[choice, :]
      pts2_choice = pts2[choice, :]
      F = eightpoint(pts1_choice, pts2_choice, M)
      ress = calc_epi_error(pts1_homo, pts2_homo, F)
      curr_num_inliner = np.sum(ress < tol)
      if curr_num_inliner > best_inlier:
          F_curr = F
          inlier_curr = (ress < tol)
          best_inlier = curr_num_inliner
  inlier_curr = inlier_curr.reshape(inlier_curr.shape[0], 1)
  indixing_array = inlier_curr.flatten()
  pts1_inlier = pts1[indixing_array]
  pts2_inlier = pts2[indixing_array]
  F = eightpoint(pts1_inlier, pts2_inlier, M)
  return F, inlier_curr

# %% Notebook cell 41
def rodrigues(r):
  '''
  Input:  r, a 3x1 vector
  Output: R, a rotation matrix
  '''

  r = np.array(r).flatten()
  I = np.eye(3)
  theta = np.linalg.norm(r)
  if theta == 0:
      return I
  else:
      U = (r/theta)[:, np.newaxis]
      Ux, Uy, Uz = r/theta
      K = np.array([[0, -Uz, Uy], [Uz, 0, -Ux], [-Uy, Ux, 0]])
      R = I * np.cos(theta) + np.sin(theta) * K + \
          (1 - np.cos(theta)) * np.matmul(U, U.T)
  return R


def invRodrigues(R):
  '''
  Input:  R, a rotation matrix
  Output: r, a 3x1 vector
  '''

  def s_half(r):
      r1, r2, r3 = r
      if np.linalg.norm(r) == np.pi and (r1 == r2 and r1 == 0 and r2 == 0 and r3 < 0) or (r1 == 0 and r2 < 0) or (r1 < 0):
          return -r
      else:
          return r

  A = (R - R.T)/2
  ro = [A[2, 1], A[0, 2], A[1, 0]]
  s = np.linalg.norm(ro)
  c = (np.sum(np.matrix(R).diagonal()) - 1)/2
  if s == 0 and c == 1:
      r = np.zeros(3)
  elif s == 0 and c == -1:
      col = np.eye(3) + R
      col_idx = np.nonzero(
          np.array(np.sum(col != 0, axis=0)).flatten())[0][0]
      v = col[:, col_idx]
      u = v/np.linalg.norm(v)
      r = s_half(u * np.pi)
  else:
      u = ro/s
      theta = np.arctan2(s, c)
      r = u * theta

  return r

# %% Notebook cell 43
def rodriguesResidual(K1, M1, p1, K2, p2, x):
  '''
  Q4.1: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
  '''
  N = p1.shape[0]
  P = x[:-6].reshape(N, 3)
  r2 = x[-6: -3]
  t2 = x[-3:]

  R2 = rodrigues(r2)
  M2 = np.hstack((R2, t2.reshape(3, 1)))
  C1 = K1 @ M1
  C2 = K2 @ M2
  P_h = np.hstack((P, np.ones((N, 1))))

  p1_hat = (C1 @ P_h.T).T
  p2_hat = (C2 @ P_h.T).T
  p1_hat = p1_hat[:, :2] / p1_hat[:, 2:3]
  p2_hat = p2_hat[:, :2] / p2_hat[:, 2:3]

  residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])

  return residuals

# %% Notebook cell 45
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
  '''
  Q4.1: Bundle adjustment.
  Input:  K1, the intrinsics of camera 1
          M1, the extrinsics of camera 1
          p1, the 2D coordinates of points in image 1
          K2,  the intrinsics of camera 2
          M2_init, the initial extrinsics of camera 1
          p2, the 2D coordinates of points in image 2
          P_init, the initial 3D coordinates of points
  Output: M2, the optimized extrinsics of camera 1
          P2, the optimized 3D coordinates of points
          o1, the starting objective function value with the initial input
          o2, the ending objective function value after bundle adjustment

  Hints:
  (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual.
      You can try different (method='..') in scipy.optimize.minimize for best results.
  '''
  obj_start = obj_end = 0

  N = p1.shape[0]
  r2 = invRodrigues(M2_init[:, :3])
  t2 = M2_init[:, 3]
  x_init = np.concatenate([P_init.reshape(-1), r2, t2])

  # we have to define objective
  def obj(x):
    return np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x) ** 2)

  obj_start = obj(x_init)
  result = scipy.optimize.minimize(obj, x_init, method='Powell')
  x = result.x
  P = x[:-6].reshape(N, 3)
  r2 = x[-6: -3]
  t2 = x[-3:]

  M2 = np.hstack((rodrigues(r2), t2.reshape(3, 1)))
  obj_end = obj(x)

  return M2, P, obj_start, obj_end

# %% Notebook cell 47
# Visualization:
np.random.seed(1)
correspondence = np.load('data/some_corresp_noisy.npz') # Loading noisy correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')
M=np.max([*im1.shape, *im2.shape])

'''
Call the ransacF function to find the fundamental matrix
Call the findM2 function to find the extrinsics of the second camera
Call the bundleAdjustment function to optimize the extrinsics and 3D points
'''
F, inlier = ransacF(pts1, pts2, M)
p1 = pts1[inlier.reshape(-1)]
p2 = pts2[inlier.reshape(-1)]
M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
M2_init, C2_init, P_init = findM2(F, p1, p2, intrinsics)
M2_final, P_final, obj_start, obj_end = bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init)

print(f"Before reprojection error: {obj_start}, After: {obj_end}")

# %% Notebook cell 48
# helper function for visualization
def plot_3D_dual(P_before, P_after, azim=70, elev=45):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    ax.view_init(azim=azim, elev=elev)
    plt.draw()

# plots the 3d points before and after BA from different viewpoints
plot_3D_dual(P_init, P_final, azim=0, elev=0)
plot_3D_dual(P_init, P_final, azim=70, elev=40)
plot_3D_dual(P_init, P_final, azim=40, elev=40)

# %% Display any remaining figures outside Jupyter
plt.show()
