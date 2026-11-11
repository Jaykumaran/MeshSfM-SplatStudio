"""Mesh generation.

A *capture* folder is the common input (any dataset can be converted to it; see datasets/):

  capture/
    images/            RGB images (rectified / pinhole)
    masks/             optional, same file names (.png): white = ignore (cars, people, sky ...)
    cameras.json       {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "width", "height",
                        "frames": [{"image": "000000.png", "T_world_cam": 4x4, "lidar": "000000.bin", "T_world_lidar": 4x4,
                                    "gps": [lat, lon, alt]}, ...]}      (poses / lidar / gps optional)
    lidar/             optional: N x 4 float32 .bin (x, y, z, intensity) or .npy / .ply / .pcd
    georef.json        optional: {"lat0", "lon0", "alt0"} of the local frame origin (ENU, z up)

Pipelines:
  lidar   (LiDAR + RGB [+ GPS poses]): scans accumulated with their poses (KISS-ICP odometry when there are none),
          coloured from the images (masked pixels never colour; points landing in masks are dropped as dynamic)
  rgb     (RGB [+ GPS poses]): COLMAP features (masks honoured) -> triangulation with the known poses
          (or full SfM without poses) -> dense MVS (CUDA) -> fused points
  points  an existing coloured point cloud (COLMAP fused.ply, LiDAR map ...)
Then mesh_from_points(): road / ground as a height field (sharp, flat, no Poisson blobs) + Poisson for everything
standing on it (trees, buildings, poles), trimmed to where there is data, colours from the points.
"""
import json, math, os, shutil, subprocess, time
import numpy as np

from .project import Project


def log(msg):
    print(msg, flush=True)


# ============================================================================================ capture IO
def load_capture(cap):
    meta = json.load(open(os.path.join(cap, 'cameras.json')))
    meta['root'] = cap
    return meta


def read_scan(path):
    if path.endswith('.bin'):
        return np.fromfile(path, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    if path.endswith('.npy'):
        return np.load(path)[:, :3].astype(np.float64)
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(path).points)


def read_mask(cap, image):
    import cv2
    f = os.path.join(cap, 'masks', os.path.splitext(image)[0] + '.png')
    if not os.path.exists(f):
        return None
    m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return m > 127


# ============================================================================================ lidar
def slam_poses(scans):
    """No LiDAR poses: KISS-SLAM (PRBonn; odometry + loop closures + pose graph), else KISS-ICP odometry."""
    try:
        from kiss_slam.config import load_config
        from kiss_slam.slam import KissSLAM
    except ImportError:
        log('kiss-slam not installed: using KISS-ICP odometry (pip install kiss-slam for loop closures)')
        return kiss_poses(scans)
    cfg = load_config(None); cfg.odometry.preprocessing.deskew = False; cfg.odometry.preprocessing.max_range = 80.0
    cfg.odometry.mapping.voxel_size = 1.0
    slam = KissSLAM(cfg)
    for i, X in enumerate(scans):
        slam.process_scan(X, np.empty((0,)))
        if i % 50 == 0:
            log(f'  KISS-SLAM {i}/{len(scans)}')
    slam.generate_new_node(); slam.local_map_graph.erase_last_local_map()
    poses, _ = slam.fine_grained_optimization()
    log(f'  KISS-SLAM: {len(poses)} poses, {len(slam.get_closures())} loop closures')
    return [np.asarray(p) for p in poses]


def kiss_poses(scans):
    """LiDAR odometry with KISS-ICP when the capture has no poses."""
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    cfg = KISSConfig(); cfg.mapping.voxel_size = 1.0; cfg.data.max_range = 80.0; cfg.data.deskew = False
    odo = KissICP(cfg); out = []
    for i, X in enumerate(scans):
        odo.register_frame(X, np.zeros(len(X)))
        out.append(np.asarray(odo.last_pose).copy())
        if i % 50 == 0:
            log(f'  KISS-ICP {i}/{len(scans)}')
    return out


def lidar_points(meta, max_range=40.0, min_range=3.0, voxel=0.05, use_masks=True):
    import cv2, open3d as o3d
    from scipy.spatial import cKDTree
    cap, fr = meta['root'], meta['frames']
    K = np.array(meta['K']); W, H = int(meta['width']), int(meta['height'])
    scans = [read_scan(os.path.join(cap, 'lidar', f['lidar'])) for f in fr]
    if any('T_world_lidar' not in f for f in fr):
        log('no LiDAR poses: running KISS-SLAM')
        poses = slam_poses(scans)
        if len(poses) != len(scans):
            log(f'  pose count {len(poses)} != scans {len(scans)}: using KISS-ICP odometry'); poses = kiss_poses(scans)
        from . import poses as PZ
        PZ.write_tum(os.path.join(cap, 'lidar_poses_estimated.tum'), poses); PZ.write_kitti(os.path.join(cap, 'lidar_poses_estimated.kitti'), poses)
        G = gps_enu_of(meta)
        if G is not None:          # put the odometry into the GPS (ENU) frame
            from .align import umeyama
            ok = np.isfinite(G[:, 0]); D = umeyama(np.array([T[:3, 3] for T in poses])[ok], G[ok], scale=False)
            poses = [D @ T for T in poses]
            res = np.linalg.norm(np.array([T[:3, 3] for T in poses])[ok] - G[ok], axis=1)
            log(f'odometry aligned to GPS: RMS {np.sqrt((res ** 2).mean()):.2f} m')
        if 'T_cam_lidar' in meta:   # camera poses from the LiDAR poses (colouring, masks)
            Tcl = np.array(meta['T_cam_lidar'])
            for f, T in zip(fr, poses):
                f.setdefault('T_world_cam', (T @ np.linalg.inv(Tcl)).tolist())
    else:
        poses = [np.array(f['T_world_lidar']) for f in fr]
    pts, dyn = [], 0
    for i, (X, T) in enumerate(zip(scans, poses)):
        r = np.linalg.norm(X[:, :2], axis=1); X = X[(r > min_range) & (r < max_range)]
        Xw = X @ T[:3, :3].T + T[:3, 3]
        if use_masks and 'T_world_cam' in fr[i]:          # dynamic objects: points that land in this frame's mask
            m = read_mask(cap, fr[i]['image'])
            if m is not None:
                Tc = np.linalg.inv(np.array(fr[i]['T_world_cam'])); Pc = Xw @ Tc[:3, :3].T + Tc[:3, 3]
                z = Pc[:, 2]; ok = z > 0.5
                u = np.full(len(Pc), -1); v = np.full(len(Pc), -1)
                u[ok] = np.round(K[0, 0] * Pc[ok, 0] / z[ok] + K[0, 2]).astype(int); v[ok] = np.round(K[1, 1] * Pc[ok, 1] / z[ok] + K[1, 2]).astype(int)
                ins = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                bad = np.zeros(len(Pc), bool); bad[ins] = m[v[ins], u[ins]]
                dyn += int(bad.sum()); Xw = Xw[~bad]
        pts.append(Xw)
    X = np.concatenate(pts)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)).voxel_down_sample(voxel)
    X = np.asarray(pc.points)
    log(f'LiDAR: {len(scans)} scans -> {len(X):,} points at {voxel * 100:.0f} cm ({dyn:,} masked dynamic points dropped)')
    C, seen = colour_from_images(meta, X, use_masks)
    return X, C, seen, np.array([T[:3, 3] for T in poses])


def colour_from_images(meta, X, use_masks=True, max_dist=30.0):
    """Colour each point from the image that sees it closest (front-facing, unmasked, not occluded by nearer points)."""
    import cv2
    cap, fr = meta['root'], meta['frames']
    K = np.array(meta['K']); W, H = int(meta['width']), int(meta['height'])
    best = np.full(len(X), np.inf); C = np.full((len(X), 3), 128, np.uint8)
    for i, f in enumerate(fr):
        if 'T_world_cam' not in f:
            continue
        Tc = np.linalg.inv(np.array(f['T_world_cam']))
        Pc = X @ Tc[:3, :3].T + Tc[:3, 3]; z = Pc[:, 2]
        cand = np.flatnonzero((z > 1.0) & (z < max_dist) & (z < best))
        if not len(cand):
            continue
        u = K[0, 0] * Pc[cand, 0] / z[cand] + K[0, 2]; v = K[1, 1] * Pc[cand, 1] / z[cand] + K[1, 2]
        ins = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1); cand, u, v = cand[ins], u[ins], v[ins]
        ui, vi = u.astype(int), v.astype(int)
        # occlusion: per 4x4 px cell keep points within 0.5 m (+5%) of the nearest
        cell = (vi // 4) * (W // 4 + 1) + ui // 4
        zmin = np.full(cell.max() + 1, np.inf); np.minimum.at(zmin, cell, z[cand])
        vis = z[cand] < zmin[cell] * 1.05 + 0.5
        m = read_mask(cap, f['image']) if use_masks else None
        if m is not None:
            vis &= ~m[vi, ui]
        cand, ui, vi = cand[vis], ui[vis], vi[vis]
        img = cv2.imread(os.path.join(cap, 'images', f['image']))[:, :, ::-1]
        C[cand] = img[vi, ui]; best[cand] = z[cand]
    seen = np.isfinite(best)
    log(f'colour: {seen.mean():.0%} of points seen by a camera')
    return C, seen


# ============================================================================================ rgb (COLMAP)
def qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat(); return np.array([w, x, y, z])


def rgb_points(meta, work, use_masks=True, colmap='colmap', fast=False, ref_step=1):
    """COLMAP: features (with masks) -> sequential matching -> triangulation with the known poses -> dense MVS."""
    cap, fr = meta['root'], meta['frames']
    K = np.array(meta['K']) if meta.get('K') is not None else None; W, H = int(meta['width']), int(meta['height'])
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, 'database.db')
    if os.path.exists(db):
        os.remove(db)

    def run(*args):
        log('  colmap ' + args[0]); subprocess.run([colmap, *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    mask_args = []
    if use_masks and os.path.isdir(os.path.join(cap, 'masks')):
        mdir = os.path.join(work, 'masks_colmap'); os.makedirs(mdir, exist_ok=True)   # COLMAP wants <image name>.png
        for f in fr:
            src = os.path.join(cap, 'masks', os.path.splitext(f['image'])[0] + '.png')
            if os.path.exists(src):
                import cv2
                cv2.imwrite(os.path.join(mdir, f['image'] + '.png'), np.where(cv2.imread(src, 0) > 127, 0, 255).astype(np.uint8))
        mask_args = ['--ImageReader.mask_path', mdir]
    cam_args = ['--ImageReader.camera_model', 'PINHOLE', '--ImageReader.camera_params', f'{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}'] if K is not None \
        else ['--ImageReader.camera_model', 'SIMPLE_RADIAL']        # intrinsics estimated by SfM
    run('feature_extractor', '--database_path', db, '--image_path', os.path.join(cap, 'images'), '--ImageReader.single_camera', '1', *cam_args, *mask_args)
    run('sequential_matcher', '--database_path', db, '--SequentialMatching.overlap', '10')
    have_poses = all('T_world_cam' in f for f in fr)
    sparse = os.path.join(work, 'sparse'); os.makedirs(sparse, exist_ok=True)
    if have_poses:
        import sqlite3
        con = sqlite3.connect(db); ids = dict((n, i) for i, n in con.execute('select image_id, name from images')); con.close()
        known = os.path.join(work, 'known'); os.makedirs(known, exist_ok=True)
        open(os.path.join(known, 'cameras.txt'), 'w').write(f'1 PINHOLE {W} {H} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n')
        with open(os.path.join(known, 'images.txt'), 'w') as f_:
            for f in fr:
                Tcw = np.linalg.inv(np.array(f['T_world_cam'])); q = qvec(Tcw[:3, :3]); t = Tcw[:3, 3]
                f_.write(f"{ids[f['image']]} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {f['image']}\n\n")
        open(os.path.join(known, 'points3D.txt'), 'w').write('')
        run('point_triangulator', '--database_path', db, '--image_path', os.path.join(cap, 'images'), '--input_path', known, '--output_path', sparse)
    else:
        run('mapper', '--database_path', db, '--image_path', os.path.join(cap, 'images'), '--output_path', sparse)
        sparse = os.path.join(sparse, '0')
        G = gps_enu_of(meta)
        if G is not None:          # similarity-align the SfM model to the GPS track (ENU metres)
            ref = os.path.join(work, 'gps_ref.txt')
            with open(ref, 'w') as f_:
                for f, g in zip(fr, G):
                    if np.isfinite(g[0]):
                        f_.write(f"{f['image']} {g[0]} {g[1]} {g[2]}\n")
            aligned = os.path.join(work, 'sparse_gps'); os.makedirs(aligned, exist_ok=True)
            run('model_aligner', '--input_path', sparse, '--output_path', aligned, '--ref_images_path', ref, '--ref_is_gps', '0',
                '--alignment_type', 'custom', '--alignment_max_error', '3.0')
            sparse = aligned; have_poses = True
    dense = os.path.join(work, 'dense')
    if os.path.exists(dense):
        shutil.rmtree(dense)
    run('image_undistorter', '--image_path', os.path.join(cap, 'images'), '--input_path', sparse, '--output_path', dense)
    if ref_step > 1:      # depth maps only for every n-th image (all images still serve as source views)
        for cfg in ('patch-match.cfg', 'fusion.cfg'):
            f = os.path.join(dense, 'stereo', cfg); L = [x for x in open(f).read().splitlines() if x.strip()]
            if cfg == 'patch-match.cfg':
                pairs = [L[i:i + 2] for i in range(0, len(L) - 1, 2)]
                open(f, 'w').write('\n'.join(x for k, p in enumerate(pairs) if k % ref_step == 0 for x in p) + '\n')
            else:
                open(f, 'w').write('\n'.join(x for k, x in enumerate(L) if k % ref_step == 0) + '\n')
    run('patch_match_stereo', '--workspace_path', dense, '--PatchMatchStereo.geom_consistency', 'false' if fast else 'true',
        '--PatchMatchStereo.max_image_size', '1600')
    run('stereo_fusion', '--workspace_path', dense, '--output_path', os.path.join(dense, 'fused.ply'), '--StereoFusion.min_num_pixels', '3' if fast else '4',
        '--input_type', 'photometric' if fast else 'geometric')
    import open3d as o3d
    pc = o3d.io.read_point_cloud(os.path.join(dense, 'fused.ply'))
    X = np.asarray(pc.points); C = (np.asarray(pc.colors) * 255).astype(np.uint8)
    log(f'dense MVS: {len(X):,} points')
    if not have_poses:
        log('note: no poses -> the result is in COLMAP\'s arbitrary frame (align it in the Align section)')
    return X, C


# ============================================================================================ meshing
def mesh_from_points(X, C, centres=None, ground_cell=0.10, poisson_depth=11, objects=True):
    """Road-aware meshing: ground / road as a height field (flat and sharp), objects on it by Poisson, trimmed."""
    import open3d as o3d
    from scipy.ndimage import gaussian_filter, binary_closing, median_filter
    from scipy.spatial import cKDTree
    from . import prep, core
    t0 = time.time()
    gz, glo, _ = prep.ground_grid(X, 0.5, pad=2.0); g_at = prep.ground_sampler(gz, glo, 0.5)
    rel = X[:, 2] - g_at(X[:, :2]); ground = (rel > -0.3) & (rel < 0.25)
    # --- ground height field
    G, Cg = X[ground], C[ground].astype(np.float64)
    lo = G[:, :2].min(0) - 1; c = ground_cell
    nx, ny = (np.ceil((G[:, :2].max(0) + 1 - lo) / c)).astype(int) + 1
    ii = ((G[:, 0] - lo[0]) / c).astype(int); jj = ((G[:, 1] - lo[1]) / c).astype(int)
    cnt = np.zeros((nx, ny)); np.add.at(cnt, (ii, jj), 1)
    zs = np.zeros((nx, ny)); np.add.at(zs, (ii, jj), G[:, 2])
    cs = np.zeros((nx, ny, 3)); np.add.at(cs, (ii, jj), Cg)
    have = cnt > 0
    mask = binary_closing(have, np.ones((5, 5)), iterations=2) | have
    Z = np.where(have, zs / np.maximum(cnt, 1), np.nan); Cc = np.where(have[..., None], cs / np.maximum(cnt, 1)[..., None], np.nan)
    Zf = core.fill_grid(Z); Zf = median_filter(Zf, 3); Zf = gaussian_filter(Zf, 1.0)
    Cf = np.stack([core.fill_grid(Cc[..., k]) for k in range(3)], -1)
    Vg, Cvg, Tg = core.grid_patch(lo, c, mask, Zf, np.clip(Cf, 0, 255), np.zeros(3))
    log(f'ground height field: {len(Tg):,} triangles at {c * 100:.0f} cm ({time.time() - t0:.0f} s)')
    parts = [(Vg.astype(np.float64), Cvg, Tg.astype(np.int64))]
    # --- objects (trees, buildings, poles): Poisson on the points above the ground
    O = X[~ground & (rel > 0)]; Co = C[~ground & (rel > 0)]
    if objects and len(O) > 1000:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(O)); pc.colors = o3d.utility.Vector3dVector(Co / 255.0)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
        if centres is not None and len(centres):        # orient normals towards the sensor path
            _, j = cKDTree(centres[:, :2]).query(O[:, :2])
            n = np.asarray(pc.normals); flip = ((centres[j] - O) * n).sum(1) < 0; n[flip] *= -1
            pc.normals = o3d.utility.Vector3dVector(n)
        m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=poisson_depth, linear_fit=True)
        dens = np.asarray(dens); m.remove_vertices_by_mask(dens < np.quantile(dens, 0.04))
        V = np.asarray(m.vertices); Tm = np.asarray(m.triangles)
        d, _ = cKDTree(O).query(V[Tm].mean(1)); keep = d < 0.12
        Vrel = V[:, 2] - g_at(V[:, :2])
        keep &= (Vrel[Tm].min(1) > -0.2)                  # nothing below the ground surface
        m.triangles = o3d.utility.Vector3iVector(Tm[keep]); m.remove_unreferenced_vertices()
        V = np.asarray(m.vertices); _, j = cKDTree(O).query(V)
        parts.append((V, Co[j], np.asarray(m.triangles).astype(np.int64)))
        log(f'objects: {len(O):,} points -> {keep.sum():,} triangles ({time.time() - t0:.0f} s)')
    Vs, Cs, Ts, off = [], [], [], 0
    for V, Cc_, Tt in parts:
        Vs.append(V); Cs.append(Cc_); Ts.append(Tt + off); off += len(V)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(Vs)), o3d.utility.Vector3iVector(np.concatenate(Ts)))
    m.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(Cs).astype(np.float64) / 255.0)
    m.compute_vertex_normals()
    return m


# ============================================================================================ entry
def run(project, body):
    """body: {pipeline: lidar|rgb|points, capture: dir (or points file for 'points'), name, masks: bool, objects: bool}"""
    import open3d as o3d
    from . import prep
    P = project if isinstance(project, Project) else Project(project)
    pipe = body['pipeline']; name = body.get('name') or pipe
    out = P.dir('gen', name)
    centres = None
    if pipe == 'points':
        meta = None
        if body.get('inputs'):            # e.g. a COLMAP model (+ images): its points (or a given dense cloud) and its camera poses
            body['capture'] = build_capture(body['inputs'], os.path.join(out, 'capture')); meta = load_capture(body['capture'])
            src = body['inputs'].get('points') or os.path.join(body['capture'], meta.get('colmap_points', 'colmap_points.ply'))
        else:
            src = body['capture']
        pc = o3d.io.read_point_cloud(src); X = np.asarray(pc.points); C = (np.asarray(pc.colors) * 255).astype(np.uint8)
        log(f'points: {len(X):,} from {os.path.basename(src)}')
        if meta is not None and all('T_world_cam' in f for f in meta['frames']):
            centres = np.array([np.array(f['T_world_cam'])[:3, 3] for f in meta['frames']])
    else:
        if body.get('inputs'):          # separate inputs -> a capture folder inside the project
            body['capture'] = build_capture(body['inputs'], os.path.join(out, 'capture'))
        meta = load_capture(body['capture'])
        if pipe == 'lidar':
            X, C, seen, centres = lidar_points(meta, use_masks=body.get('masks', True))
            if body.get('only_seen', True):
                X, C = X[seen], C[seen]
        elif pipe == 'rgb':
            X, C = rgb_points(meta, os.path.join(out, 'colmap'), use_masks=body.get('masks', True), fast=body.get('fast', False), ref_step=int(body.get('ref_step', 1)))
        else:
            raise ValueError(pipe)
        if centres is None and all('T_world_cam' in f for f in meta['frames']):
            centres = np.array([np.array(f['T_world_cam'])[:3, 3] for f in meta['frames']])
    pcf = os.path.join(out, 'points.ply')
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)); pc.colors = o3d.utility.Vector3dVector(C / 255.0)
    o3d.io.write_point_cloud(pcf, pc)
    m = mesh_from_points(X, C, centres, objects=body.get('objects', True))
    mf = os.path.join(out, 'mesh.ply'); o3d.io.write_triangle_mesh(mf, m)
    log(f'mesh written: {len(m.triangles):,} triangles')
    P.data.setdefault('generated', {})[name] = dict(pipeline=pipe, capture=body.get('capture'), points=pcf, mesh=mf, created=time.strftime('%Y-%m-%d %H:%M:%S'))
    P.save()
    if body.get('add_layers', True):
        P.add_layer('mesh', mf, name=f'{name} mesh')
        P.add_layer('points', pcf, name=f'{name} points', visible=False)
    if meta is not None and centres is not None and not os.path.exists(os.path.join(P.mesh_dir, 'path.npz')):
        Pp, arc, F = prep.path_from_points(centres - np.array([0, 0, body.get('sensor_height', 1.65)]))
        prep.save_path(P, Pp, arc, F, f'{pipe} poses')
    if meta is not None and os.path.exists(os.path.join(body['capture'], 'georef.json')) and not os.path.exists(os.path.join(P.root, 'georef.json')):
        shutil.copy(os.path.join(body['capture'], 'georef.json'), os.path.join(P.root, 'georef.json'))
    if body.get('prepare', True):
        prep.prepare(P)
    return dict(mesh=mf, points=pcf, triangles=int(len(m.triangles)), points_n=int(len(X)))


# ============================================================================================ separate inputs -> capture
def _read_pose_table(path):
    """Pose file -> {name: 4x4 T_world_X}. Formats: our cameras.json; text/CSV lines 'name tx ty tz qw qx qy qz' or
    'name' + 16 numbers (row-major 4x4); KITTI-odometry style lines of 12 numbers (row-major 3x4, matched by order)."""
    from scipy.spatial.transform import Rotation
    if path.endswith('.json'):
        J = json.load(open(path))
        return {f['image']: np.array(f['T_world_cam']) for f in J['frames'] if 'T_world_cam' in f}
    out, k = {}, 0
    for line in open(path):
        t = line.replace(',', ' ').split()
        if not t or t[0].startswith('#'):
            continue
        try:
            float(t[0]); name, v = f'#{k}', [float(x) for x in t]
        except ValueError:
            name, v = t[0], [float(x) for x in t[1:]]
        T = np.eye(4)
        if len(v) == 7:
            T[:3, :3] = Rotation.from_quat([v[4], v[5], v[6], v[3]]).as_matrix(); T[:3, 3] = v[:3]
        elif len(v) == 12:
            T[:3] = np.array(v).reshape(3, 4)
        elif len(v) == 16:
            T = np.array(v).reshape(4, 4)
        else:
            continue
        out[name] = T; k += 1
    return out


def _read_gps(path):
    """GPS CSV / TXT: 'name, lat, lon, alt' per line (header allowed) -> ({name: (lat, lon, alt)}, ordered list)."""
    out = {}
    for line in open(path):
        t = [x.strip() for x in line.replace(';', ',').replace('\t', ',').split(',') if x.strip()]
        if len(t) < 3:
            t = line.split()
        try:
            vals = [float(x) for x in t[-3:]]
        except ValueError:
            continue
        name = t[0] if len(t) >= 4 else f'#{len(out)}'
        out[name] = vals
    return out


def enu(gps):
    """lat / lon / alt -> local east-north-up metres (origin = first fix)."""
    g = np.asarray(gps, float); lat0, lon0, alt0 = g[0]
    a, e2 = 6378137.0, 6.69437999014e-3; s = math.sin(math.radians(lat0))
    Rm = a * (1 - e2) / (1 - e2 * s * s) ** 1.5; Rn = a / math.sqrt(1 - e2 * s * s)
    E = np.radians(g[:, 1] - lon0) * Rn * math.cos(math.radians(lat0)); N = np.radians(g[:, 0] - lat0) * Rm
    return np.c_[E, N, g[:, 2] - alt0], (lat0, lon0, alt0)


def _match(names, table):
    """Look up per-image values by file name, stem, or order."""
    if not table:
        return [None] * len(names)
    stems = {os.path.splitext(k)[0]: v for k, v in table.items()}
    vals = list(table.values())
    out = []
    for i, n in enumerate(names):
        v = table.get(n, stems.get(os.path.splitext(n)[0]))
        if v is None and all(k.startswith('#') for k in table):
            v = vals[min(len(vals) - 1, int(round(i * (len(vals) - 1) / max(1, len(names) - 1))))]
        out.append(v)
    return out


def build_capture(spec, out):
    """spec: images (dir, required), masks (dir), lidar (dir), poses (camera poses file), lidar_poses (file),
    gps (file), intrinsics ('fx,fy,cx,cy'), T_cam_lidar (file with 16 or 12 numbers: LiDAR -> camera)."""
    import cv2
    os.makedirs(out, exist_ok=True)
    for k, sub in (('images', 'images'), ('masks', 'masks'), ('lidar', 'lidar')):
        dst = os.path.join(out, sub)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst) if os.path.islink(dst) else shutil.rmtree(dst)
        if spec.get(k):
            os.symlink(os.path.abspath(spec[k]), dst)
    names = sorted(f for f in os.listdir(os.path.join(out, 'images')) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    h, w = cv2.imread(os.path.join(out, 'images', names[0])).shape[:2]
    scans = sorted(f for f in os.listdir(os.path.join(out, 'lidar'))) if spec.get('lidar') else []
    from . import poses as PZ
    stamps = [float(x) for x in open(spec['timestamps']).read().split()] if spec.get('timestamps') else None
    cmodel = None
    if spec.get('colmap'):                       # a COLMAP sparse model: intrinsics + camera poses (+ its points)
        cmodel = PZ.read_colmap(spec['colmap']); spec.setdefault('intrinsics', ','.join(str(v) for v in (cmodel['K'][0][0], cmodel['K'][1][1], cmodel['K'][0][2], cmodel['K'][1][2])))
    cam = [None] * len(names)
    if spec.get('poses'):
        P_, fmt = PZ.read(spec['poses']); cam = PZ.match(names, P_, stamps); log(f'camera poses: {len(P_)} ({fmt})')
    elif cmodel:
        cam = PZ.match(names, [dict(key=k, T=v, t=None) for k, v in cmodel['poses'].items()]); log(f"camera poses: {len(cmodel['poses'])} (colmap)")
    lp = [None] * len(scans)
    if spec.get('lidar_poses') and scans:
        P_, fmt = PZ.read(spec['lidar_poses']); lstamps = [float(x) for x in open(spec['lidar_timestamps']).read().split()] if spec.get('lidar_timestamps') else None
        lp = PZ.match(scans, P_, lstamps); log(f'LiDAR poses: {len(P_)} ({fmt})')
    Tcl = None
    if spec.get('T_cam_lidar'):
        v = [float(x) for x in open(spec['T_cam_lidar']).read().replace(',', ' ').split()]
        Tcl = np.eye(4); Tcl[:len(v) // 4] = np.array(v).reshape(-1, 4)
    gps = _match(names, _read_gps(spec['gps'])) if spec.get('gps') else [None] * len(names)
    K = None
    if spec.get('intrinsics'):
        fx, fy, cx, cy = [float(x) for x in str(spec['intrinsics']).replace(' ', ',').split(',') if x]
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    elif spec.get('poses', '').endswith('.json'):
        K = json.load(open(spec['poses'])).get('K')
    frames = []
    for i, n in enumerate(names):
        f = dict(image=n)
        if scans:
            j = min(len(scans) - 1, int(round(i * (len(scans) - 1) / max(1, len(names) - 1))))
            f['lidar'] = scans[j]
            if lp[j] is not None:
                f['T_world_lidar'] = np.asarray(lp[j]).tolist()
                if cam[i] is None and Tcl is not None:
                    cam[i] = np.asarray(lp[j]) @ np.linalg.inv(Tcl)
        if cam[i] is not None:
            f['T_world_cam'] = np.asarray(cam[i]).tolist()
        if gps[i] is not None:
            f['gps'] = list(gps[i])
        frames.append(f)
    meta = dict(K=K, width=int(w), height=int(h), frames=frames, source='built from separate inputs')
    if cmodel is not None and len(cmodel['points']):
        import open3d as o3d
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cmodel['points'])); pc.colors = o3d.utility.Vector3dVector(cmodel['colors'] / 255.0)
        o3d.io.write_point_cloud(os.path.join(out, 'colmap_points.ply'), pc); meta['colmap_points'] = 'colmap_points.ply'
    if Tcl is not None:
        meta['T_cam_lidar'] = Tcl.tolist()
    if any('gps' in f for f in frames):
        G = [f['gps'] for f in frames if 'gps' in f]; _, o = enu(G)
        json.dump(dict(lat0=o[0], lon0=o[1], alt0=o[2], frame='local ENU (metres), z up'), open(os.path.join(out, 'georef.json'), 'w'))
    json.dump(meta, open(os.path.join(out, 'cameras.json'), 'w'), indent=1)
    log(f'capture: {len(names)} images ({w}x{h}), {len(scans)} LiDAR scans, camera poses {sum(c is not None for c in cam)}, '
        f'LiDAR poses {sum(p is not None for p in lp)}, GPS {sum(g is not None for g in gps)}, intrinsics {"yes" if K else "estimated"}')
    return out


def gps_enu_of(meta):
    """Per-frame ENU positions from the capture's GPS (None where missing)."""
    fr = meta['frames']; have = [i for i, f in enumerate(fr) if 'gps' in f]
    if len(have) < 3:
        return None
    P, _ = enu([fr[i]['gps'] for i in have])
    out = np.full((len(fr), 3), np.nan); out[have] = P
    return out
