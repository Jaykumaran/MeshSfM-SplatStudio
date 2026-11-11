"""Pose files in the common formats -> list of (key, T 4x4, timestamp or None).

  TUM          'timestamp tx ty tz qx qy qz qw'                     (.tum / .txt with 8 numbers per row)
  KITTI        12 numbers per row: the 3x4 [R | t] (row-major)       (.kitti / .txt), optional times.txt next to it
  COLMAP       a sparse model folder (cameras / images / points3D .bin or .txt): poses, intrinsics, points
  table        'name tx ty tz qw qx qy qz'  or  'name' + 16 numbers (4x4 row-major)  or  'name' + 12 numbers
  cameras.json this project's capture format
All poses are sensor -> world (COLMAP's world -> camera is inverted)."""
import json, os, struct
import numpy as np
from scipy.spatial.transform import Rotation


def _rt(q_xyzw, t):
    T = np.eye(4); T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix(); T[:3, 3] = t; return T


def read(path):
    """-> (list of dict(key, T, t), format name)."""
    if os.path.isdir(path):
        m = read_colmap(path); return [dict(key=k, T=v, t=None) for k, v in m['poses'].items()], 'colmap'
    if path.endswith('.json'):
        J = json.load(open(path))
        return [dict(key=f['image'], T=np.array(f['T_world_cam']), t=None) for f in J['frames'] if 'T_world_cam' in f], 'cameras.json'
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        tok = s.replace(',', ' ').split()
        try:
            vals = [float(x) for x in tok]; name = None
        except ValueError:
            name = tok[0]
            try:
                vals = [float(x) for x in tok[1:]]
            except ValueError:
                continue                                   # header line
        rows.append((name, vals))
    if not rows:
        return [], 'empty'
    n = len(rows[0][1]); named = rows[0][0] is not None
    out = []
    times = None
    tf = os.path.join(os.path.dirname(path), 'times.txt')
    if not named and n == 12 and os.path.exists(tf):
        times = [float(x) for x in open(tf).read().split()]
    for i, (name, v) in enumerate(rows):
        if not named and n == 8:                       # TUM
            out.append(dict(key=f'#{i}', T=_rt(v[4:8], v[1:4]), t=v[0]))
        elif not named and n == 12:                    # KITTI
            T = np.eye(4); T[:3] = np.array(v).reshape(3, 4); out.append(dict(key=f'#{i}', T=T, t=times[i] if times and i < len(times) else None))
        elif not named and n == 16:
            out.append(dict(key=f'#{i}', T=np.array(v).reshape(4, 4), t=None))
        elif named and len(v) == 7:                    # name tx ty tz qw qx qy qz
            out.append(dict(key=name, T=_rt([v[4], v[5], v[6], v[3]], v[:3]), t=None))
        elif named and len(v) == 8:                    # name + TUM row
            out.append(dict(key=name, T=_rt(v[4:8], v[1:4]), t=v[0]))
        elif named and len(v) in (12, 16):
            T = np.eye(4); T[:len(v) // 4] = np.array(v).reshape(-1, 4); out.append(dict(key=name, T=T, t=None))
    fmt = 'tum' if (not named and n == 8) else 'kitti' if (not named and n == 12) else 'table'
    return out, fmt


def match(names, poses, stamps=None):
    """Per item (image / scan) the pose: by file name or stem, else by timestamp (nearest; item stamps from a list or
    numeric file names), else by order (index-proportional)."""
    if not poses:
        return [None] * len(names)
    by = {p['key']: p['T'] for p in poses}; by.update({os.path.splitext(k)[0]: v for k, v in list(by.items())})
    got = [by.get(n, by.get(os.path.splitext(n)[0])) for n in names]
    if all(g is not None for g in got):
        return got
    pt = np.array([p['t'] if p['t'] is not None else np.nan for p in poses])
    if stamps is None:
        try:
            stamps = [float(os.path.splitext(n)[0]) for n in names]
        except ValueError:
            stamps = None
    if stamps is not None and np.isfinite(pt).all() and len(stamps) == len(names):
        st = np.asarray(stamps, float); scale = 1.0
        if np.median(st) > 1e14 and np.median(pt) < 1e11:     # names in ns, poses in s
            scale = 1e-9
        j = np.clip(np.searchsorted(pt, st * scale), 1, len(pt) - 1)
        j = np.where(np.abs(pt[j - 1] - st * scale) < np.abs(pt[j] - st * scale), j - 1, j)
        return [poses[k]['T'] for k in j]
    return [poses[min(len(poses) - 1, int(round(i * (len(poses) - 1) / max(1, len(names) - 1))))]['T'] for i in range(len(names))]


def write_tum(path, Ts, times=None):
    with open(path, 'w') as f:
        for i, T in enumerate(Ts):
            q = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat(); t = np.asarray(T)[:3, 3]
            f.write(f'{times[i] if times is not None else float(i):.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n')


def write_kitti(path, Ts):
    with open(path, 'w') as f:
        for T in Ts:
            f.write(' '.join(f'{v:.8e}' for v in np.asarray(T)[:3].ravel()) + '\n')


# ------------------------------------------------------------------------------------------- COLMAP
CAM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
CAM_NAMES = {0: 'SIMPLE_PINHOLE', 1: 'PINHOLE', 2: 'SIMPLE_RADIAL', 3: 'RADIAL', 4: 'OPENCV', 5: 'OPENCV_FISHEYE', 6: 'FULL_OPENCV', 7: 'FOV',
             8: 'SIMPLE_RADIAL_FISHEYE', 9: 'RADIAL_FISHEYE', 10: 'THIN_PRISM_FISHEYE'}


def _K(model, p):
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE', 'FOV'):
        return [[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]]
    return [[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]]


def read_colmap(d):
    """COLMAP sparse model (.bin or .txt) -> dict(K, width, height, model, poses {image name: T_world_cam}, points (n,3), colors (n,3))."""
    cams, poses, pts, cols = {}, {}, [], []
    if os.path.exists(os.path.join(d, 'cameras.bin')):
        with open(os.path.join(d, 'cameras.bin'), 'rb') as f:
            for _ in range(struct.unpack('<Q', f.read(8))[0]):
                cid, mid, w, h = struct.unpack('<iiQQ', f.read(24)); n = CAM_PARAMS[mid]
                cams[cid] = (CAM_NAMES[mid], w, h, struct.unpack('<' + 'd' * n, f.read(8 * n)))
        with open(os.path.join(d, 'images.bin'), 'rb') as f:
            for _ in range(struct.unpack('<Q', f.read(8))[0]):
                iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack('<idddddddi', f.read(64))
                name = b''
                while True:
                    c = f.read(1)
                    if c == b'\0':
                        break
                    name += c
                n2 = struct.unpack('<Q', f.read(8))[0]; f.read(24 * n2)
                Tcw = _rt([qx, qy, qz, qw], [tx, ty, tz]); poses[name.decode()] = (np.linalg.inv(Tcw), cid)
        with open(os.path.join(d, 'points3D.bin'), 'rb') as f:
            for _ in range(struct.unpack('<Q', f.read(8))[0]):
                v = struct.unpack('<QdddBBBd', f.read(43)); pts.append(v[1:4]); cols.append(v[4:7])
                tl = struct.unpack('<Q', f.read(8))[0]; f.read(8 * tl)
    else:
        for line in open(os.path.join(d, 'cameras.txt')):
            if line.strip() and not line.startswith('#'):
                t = line.split(); cams[int(t[0])] = (t[1], int(t[2]), int(t[3]), [float(x) for x in t[4:]])
        L = [l for l in open(os.path.join(d, 'images.txt')) if l.strip() and not l.startswith('#')]
        for l in L[0::2]:
            t = l.split(); qw, qx, qy, qz, tx, ty, tz = map(float, t[1:8])
            poses[t[9]] = (np.linalg.inv(_rt([qx, qy, qz, qw], [tx, ty, tz])), int(t[8]))
        for line in open(os.path.join(d, 'points3D.txt')):
            if line.strip() and not line.startswith('#'):
                t = line.split(); pts.append([float(x) for x in t[1:4]]); cols.append([int(x) for x in t[4:7]])
    cid0 = next(iter(cams)); model, w, h, p = cams[cid0]
    if len(cams) > 1:
        print(f'note: {len(cams)} cameras in the model; using camera {cid0} ({model}) intrinsics for all', flush=True)
    if model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
        print(f'note: camera model {model}: lens distortion is ignored (use undistorted images, e.g. COLMAP image_undistorter)', flush=True)
    return dict(K=_K(model, p), width=w, height=h, model=model, poses={k: v[0] for k, v in poses.items()},
                points=np.array(pts, float).reshape(-1, 3), colors=np.array(cols, np.uint8).reshape(-1, 3))
