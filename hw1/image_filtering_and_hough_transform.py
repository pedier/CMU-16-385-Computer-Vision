"""CMU 16-385: Computer Vision — Image Filtering and Hough Transform.

Python companion to Image Filtering and Hough Transform.ipynb.
Run this script to execute the notebook exercises in their original order.
Keep the complete repository together so bundled inputs remain available.
"""

# %% Local setup
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT / "hw1")
Path("results").mkdir(exist_ok=True)
print("Working directory:", Path.cwd())

# %% Notebook cell 9
import numpy as np

def myImageFilter(img0, h):
  img0 = np.asarray(img0, dtype=np.float64)
  h = np.asarray(h, dtype=np.float64)
  img_height, img_width = img0.shape
  h_height, h_width = h.shape   # both should be odd

  # first, let's pad the image, using edge mode which I prefer
  # the size of the padding depends on the size of the kernel
  pad_height = h_height // 2
  pad_width = h_width // 2
  padded = np.pad(img0, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

  # convolution requires flipping of kernel
  h_flipped = np.flip(np.flip(h, axis=0), axis=1)

  img1 = np.zeros((img_height, img_width), dtype=np.float64)

  # instead of looping over the image, let's loop over the kernel
  for i in range(h_height):
    for j in range(h_width):
      img1 += h_flipped[i, j] * padded[i:i+img_height, j:j+img_width]

  return img1

# %% Notebook cell 11
import numpy as np
from scipy.signal.windows import gaussian # For gaussian kernel

def myEdgeFilter(img0, sigma):
  img0 = np.asarray(img0, dtype=np.float64)
  sigma = float(sigma)

  hsize = int(2 * np.ceil(3 * sigma) + 1)
  g_1d = gaussian(hsize, sigma)
  g_2d = np.outer(g_1d, g_1d)
  g_nor = g_2d / np.sum(g_2d)

  smoothed = myImageFilter(img0, g_nor)

  sb_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
  sb_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
  imgx = myImageFilter(smoothed, sb_x)
  imgy = myImageFilter(smoothed, sb_y)

  mag = np.sqrt(imgx ** 2 + imgy ** 2)
  dir = (180 + np.rad2deg(np.arctan2(imgy, imgx))) % 180

  # pad a circle of zeros
  mag_padded = np.pad(mag, ((1, 1), (1, 1)), mode='constant', constant_values=0)

  Nnbr = mag_padded[:-2, 1:-1]
  Snbr = mag_padded[2:, 1:-1]
  Wnbr = mag_padded[1:-1, :-2]
  Enbr = mag_padded[1:-1, 2:]
  NWnbr = mag_padded[:-2, :-2]
  NEnbr = mag_padded[:-2, 2:]
  SWnbr = mag_padded[2:, :-2]
  SEnbr = mag_padded[2:, 2:]

  # partition into intervals
  in_0 = ((dir < 22.5) | (dir >= 157.5))
  in_45 = ((dir >= 22.5) & (dir < 67.5))
  in_90 = ((dir >= 67.5) & (dir < 112.5))
  in_135 = ((dir >= 112.5) & (dir < 157.5))

  keep_0 = (in_0 & (mag >= Wnbr) & (mag >= Enbr))
  keep_45 = (in_45 & (mag >= NEnbr) & (mag >= SWnbr))
  keep_90 = (in_90 & (mag >= Nnbr) & (mag >= Snbr))
  keep_135 = (in_135 & (mag >= NWnbr) & (mag >= SEnbr))

  kept = (keep_0 | keep_45 | keep_90 | keep_135)
  img1 = np.where(kept, mag, 0.0)
  return img1

# %% Notebook cell 13
import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
  Im_height, Im_width = Im.shape
  rhoRes = float(rhoRes)
  thetaRes = float(thetaRes)

  farthest = np.sqrt((Im_height - 1) ** 2 + (Im_width - 1) ** 2)
  numrho = int(np.ceil(farthest / rhoRes)) + 1

  rhoScale = np.arange(numrho, dtype=np.float64) * rhoRes

  thetaScale = np.arange(0.0, 2.0 * np.pi, thetaRes, dtype=np.float64)

  img_hough = np.zeros((np.size(rhoScale), np.size(thetaScale)), dtype=np.float64)

  edge_y, edge_x = np.where(Im > 0)
  sintheta, costheta = np.sin(thetaScale), np.cos(thetaScale)

  for i in range(np.size(thetaScale)):
    rho = edge_x * costheta[i] + edge_y * sintheta[i]
    nonneg = (rho >= 0)

    # identify, then vote
    rho_ind = np.floor(rho[nonneg] / rhoRes + 0.5).astype(np.int64)
    img_hough[:, i] = np.bincount(rho_ind, minlength=np.size(rhoScale))

  return img_hough, rhoScale, thetaScale

# %% Notebook cell 15
import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
  H = np.asarray(H, dtype=np.float64)
  nms = 5
  kernel = np.ones((nms, nms), dtype=np.uint8)
  kernel[nms//2, nms//2] = 0  # we have to exclude the center
  H_dilated = cv2.dilate(H, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

  peaks = ((H > 0) & (H_dilated < H)) # greater than all neighbors - keep
  peaks_coor = np.argwhere(peaks)
  peaks_val = H[peaks]

  # need to sort, take the first nLines
  sorted_indices = np.argsort(peaks_val)[::-1][:nLines]
  sorted_peaks_coor = peaks_coor[sorted_indices]

  rhos = sorted_peaks_coor[:, 0].astype(np.int64)
  thetas = sorted_peaks_coor[:, 1].astype(np.int64)
  return rhos, thetas

# %% Notebook cell 17
import cv2
import numpy as np
import os

datadir    = 'data'      # the directory containing the images
resultsdir = 'results'   # the directory for dumping results

# parameters
sigma     = 2
threshold = 0.03
rhoRes    = 2
thetaRes  = np.pi / 90
nLines    = 15
# end of parameters

for file in os.listdir(datadir):
    if file.endswith('.jpg'):

        file = os.path.splitext(file)[0]

        # read in images
        img = cv2.imread('%s/%s.jpg' % (datadir, file))

        if (img.ndim == 3):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = np.float32(img) / 255

        # actual Hough line code function calls
        img_edge = myEdgeFilter(img, sigma)
        img_threshold = np.float32(img_edge > threshold)
        [img_hough, rhoScale, thetaScale] = myHoughTransform(img_threshold, \
                                                             rhoRes, thetaRes)
        [rhos, thetas] = myHoughLines(img_hough, nLines)

        lines = cv2.HoughLinesP(np.uint8(255 * img_threshold), rhoRes, thetaRes, \
                                50, minLineLength = 20, maxLineGap = 5)

        # everything below here just saves the outputs to files
        fname = '%s/%s_01edge.png' % (resultsdir, file)
        cv2.imwrite(fname, 255 * np.sqrt(img_edge / img_edge.max()))

        fname = '%s/%s_02threshold.png' % (resultsdir, file)
        cv2.imwrite(fname, 255 * img_threshold)

        fname = '%s/%s_03hough.png' % (resultsdir, file)
        cv2.imwrite(fname, 255 * img_hough / img_hough.max())

        fname = '%s/%s_04lines.png' % (resultsdir, file)
        img_lines = np.dstack([img,img,img])

        # display line results from myHoughLines function in red
        for k in np.arange(nLines):
            a = np.cos(thetaScale[thetas[k]])
            b = np.sin(thetaScale[thetas[k]])

            x0 = a*rhoScale[rhos[k]]
            y0 = b*rhoScale[rhos[k]]

            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))

            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img_lines,(x1,y1),(x2,y2),(0,0,255),1)

        # display line segment results from cv2.HoughLinesP in green, to serve
        # as a comparison
        for coords in ([] if lines is None else lines.reshape(-1, 4)):
            cv2.line(img_lines, (int(coords[0]), int(coords[1])),
                     (int(coords[2]), int(coords[3])), (0, 255, 0), 1)

        cv2.imwrite(fname, 255 * img_lines)
