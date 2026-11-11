"""Capture -> COLMAP text model, ready for any Gaussian-splatting trainer (train it wherever you like):

  OUT/images/          (symlinks to the capture images)
  OUT/sparse/0/        cameras.txt (PINHOLE), images.txt (the capture's poses), points3D.txt (initial points)

The initial points come from a coloured point cloud (e.g. the LiDAR map of the lidar pipeline); poses stay in the
capture frame, so the trained splat lands in the same frame as the mesh (no alignment needed).

  python -m meshsfm.export_colmap CAPTURE POINTS.ply OUT [--max-points 600000]
"""
import argparse, json, os
import numpy as np
from scipy.spatial.transform import Rotation


def export(capture, points, out, max_points=600_000):
    import open3d as o3d
    meta = json.load(open(os.path.join(capture, 'cameras.json')))
    K = np.array(meta['K']); W, H = int(meta['width']), int(meta['height'])
    sp = os.path.join(out, 'sparse', '0'); os.makedirs(sp, exist_ok=True)
    im = os.path.join(out, 'images'); os.makedirs(im, exist_ok=True)
    open(os.path.join(sp, 'cameras.txt'), 'w').write(f'1 PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n')
    n = 0
    with open(os.path.join(sp, 'images.txt'), 'w') as f:
        for i, fr in enumerate(meta['frames']):
            if 'T_world_cam' not in fr:
                continue
            Tcw = np.linalg.inv(np.array(fr['T_world_cam']))
            x, y, z, w = Rotation.from_matrix(Tcw[:3, :3]).as_quat(); t = Tcw[:3, 3]
            f.write(f"{i + 1} {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} 1 {fr['image']}\n\n")
            dst = os.path.join(im, fr['image'])
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(os.path.join(capture, 'images', fr['image'])), dst)
            n += 1
    pc = o3d.io.read_point_cloud(points); X = np.asarray(pc.points); C = (np.asarray(pc.colors) * 255).round().astype(int)
    if len(X) > max_points:
        s = np.random.default_rng(0).choice(len(X), max_points, replace=False); X, C = X[s], C[s]
    with open(os.path.join(sp, 'points3D.txt'), 'w') as f:
        for j, (p, c) in enumerate(zip(X, C)):
            f.write(f'{j + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n')
    print(f'COLMAP model: {n} images, {len(X):,} points -> {out}', flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture'); ap.add_argument('points'); ap.add_argument('out'); ap.add_argument('--max-points', type=int, default=600_000)
    a = ap.parse_args(); export(a.capture, a.points, a.out, a.max_points)
