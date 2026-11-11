# Third-party software and data

Nothing below is bundled with this repository; it is installed or downloaded by the user.

| Component | Use | License |
|---|---|---|
| [NumPy](https://numpy.org), [SciPy](https://scipy.org), [Matplotlib](https://matplotlib.org) | arrays, sparse solves, geometry | BSD-3-Clause |
| [Open3D](http://www.open3d.org) | mesh / point-cloud IO, Poisson reconstruction, ICP | MIT |
| [OpenCV](https://opencv.org) | images, orthophoto tiles | Apache-2.0 |
| [KISS-SLAM](https://github.com/PRBonn/kiss-slam) / [KISS-ICP](https://github.com/PRBonn/kiss-icp) | LiDAR poses when a capture has none | MIT |
| [laspy](https://github.com/laspy/laspy) | .las / .laz point clouds | BSD-2-Clause |
| [PyTorch](https://pytorch.org) / [torchvision](https://github.com/pytorch/vision) | optional auto-masks (DeepLabV3-MobileNetV3, Pascal VOC classes) | BSD-3-Clause |
| [Roboto](https://fonts.google.com/specimen/Roboto) | UI font (Google Fonts, falls back to the system font offline) | Apache-2.0 / OFL |
| [COLMAP](https://colmap.github.io) | RGB pipeline: features, triangulation, dense MVS | BSD-3-Clause |
| [KITTI raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) | demo data (downloaded by `scripts/get_kitti_demo.sh`) | CC BY-NC-SA 3.0 |

The Gaussian-splat renderer (`meshsfm/web/splat.js`) is an original WebGL2 implementation of the standard
EWA splatting of 3D Gaussians (Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, 2023).
