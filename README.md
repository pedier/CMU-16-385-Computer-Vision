# CMU 16-385: Computer Vision

Fall 2026 coursework: three programming projects plus the HW0 Python/NumPy introduction. Each notebook is bundled with its corresponding inputs and saved results.

| Folder | Notebook | Bundled inputs | Colab |
| --- | --- | --- | --- |
| `hw0` | [NumPy and visualization](hw0/Numpy%20and%20Visualize.ipynb) | `numpy/442solutions/`, `visualize/mysterydata/` | [Open](https://colab.research.google.com/github/pedier/CMU-16-385-Computer-Vision/blob/main/hw0/Numpy%20and%20Visualize.ipynb) |
| `hw1` | [Image filtering and Hough transform](hw1/Image%20Filtering%20and%20Hough%20Transform.ipynb) | `data/` | [Open](https://colab.research.google.com/github/pedier/CMU-16-385-Computer-Vision/blob/main/hw1/Image%20Filtering%20and%20Hough%20Transform.ipynb) |
| `hw2` | [Augmented reality with planar homographies](hw2/Augmented%20Reality%20with%20Planar%20Homographies.ipynb) | `data/`, `results/brief_rot_test.pkl` | [Open](https://colab.research.google.com/github/pedier/CMU-16-385-Computer-Vision/blob/main/hw2/Augmented%20Reality%20with%20Planar%20Homographies.ipynb) |
| `hw3` | [3D reconstruction](hw3/3D%20Reconstruction.ipynb) | `data/` images, correspondences, and camera intrinsics | [Open](https://colab.research.google.com/github/pedier/CMU-16-385-Computer-Vision/blob/main/hw3/3D%20Reconstruction.ipynb) |

## Run locally

Clone or download the **whole repository**. In a Python 3.12 virtual environment, run from its root:

```sh
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open a notebook and run its setup cell first. It selects the correct working directory, so the existing relative data paths work. Keep the `hw0`–`hw3` directory structure together.

## Google Colab

Use a Colab link in the table. The setup cell clones this repository and installs the requirements automatically; no Google Drive mount or separate ZIP download is needed.

All 157 Drive assets are included, including the HW0 reference arrays and the Project 2 rotation-test cache. `checksums.sha256` records their original SHA-256 hashes. PDF export is optional and separate from execution.

Portability changes are limited to repository-relative setup, optional PDF export, standard notebook metadata, the supplied BRIEF scalar-index helper, and OpenCV line drawing. The original exercise algorithms are retained. HW0's `t2` reference check reports a complex-versus-real dtype mismatch with the tested NumPy version.
