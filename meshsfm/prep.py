"""Preparation: from the project's layers to what the studio streams and edits.

  prepare(project)  = path (from poses / or the data's main axis) -> ground grid -> mesh chunks -> orthophoto
                      -> splat tiles. Re-run after adding layers or changing a layer transform.
"""
import json, math, os, time
import numpy as np

from . import core
from .project import apply, Project

SH0 = 0.28209479


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------- readers
def read_mesh(path, T):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    V = apply(T, np.asarray(m.vertices, np.float64))
    C = (np.asarray(m.vertex_colors) * 255).round().astype(np.uint8) if m.has_vertex_colors() else np.full((len(V), 3), 150, np.uint8)
    return V, C, np.asarray(m.triangles).astype(np.int64)


def read_points(path, T):
    """Coloured points: .ply / .pcd / .xyz (open3d), .las/.laz (laspy), .npz (xyz, rgb)."""
    p = path.lower()
    if p.endswith('.npz'):
        z = np.load(path); X = z['xyz']; C = z['rgb'] if 'rgb' in z.files else np.full((len(X), 3), 150, np.uint8)
    elif p.endswith(('.las', '.laz')):
        import laspy
        f = laspy.read(path); X = np.c_[f.x, f.y, f.z]
        C = (np.c_[f.red, f.green, f.blue] / (256.0 if np.max(f.red) > 255 else 1.0)).astype(np.uint8) if hasattr(f, 'red') else np.full((len(X), 3), 150, np.uint8)
    else:
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path); X = np.asarray(pc.points)
        C = (np.asarray(pc.colors) * 255).astype(np.uint8) if pc.has_colors() else np.full((len(X), 3), 150, np.uint8)
    return apply(T, np.asarray(X, np.float64)), np.asarray(C, np.uint8)


def read_splat(path, T, max_n=None):
    """3DGS .ply -> internal frame: positions, colours (DC), opacity, log-scales (+log s), rotations (quaternions w,x,y,z)."""
    a, props = core.load_ply_memmap(path)
    ix = {k: i for i, k in enumerate(props)}
    n = len(a)
    sel = np.arange(n) if not max_n or n <= max_n else np.sort(np.random.default_rng(0).choice(n, max_n, replace=False))
    A = np.asarray(a[sel])
    T = np.asarray(T, float); R = T[:3, :3]; s = np.cbrt(abs(np.linalg.det(R))); Rn = R / s
    X = A[:, [ix['x'], ix['y'], ix['z']]] @ R.T + T[:3, 3]
    rgb = np.clip(0.5 + SH0 * A[:, [ix['f_dc_0'], ix['f_dc_1'], ix['f_dc_2']]], 0, 1)
    op = 1 / (1 + np.exp(-A[:, ix['opacity']]))
    sc = A[:, [ix['scale_0'], ix['scale_1'], ix['scale_2']]] + np.log(s)
    q = A[:, [ix['rot_0'], ix['rot_1'], ix['rot_2'], ix['rot_3']]].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    from scipy.spatial.transform import Rotation
    rr = Rotation.from_matrix(Rn) * Rotation.from_quat(q[:, [1, 2, 3, 0]])
    qq = rr.as_quat()[:, [3, 0, 1, 2]]
    return X, rgb, op, sc, qq


# ------------------------------------------------------------------------------------------- path
def path_from_points(xyz, step=1.0):
    """Ordered polyline -> path arrays (P, arc, F), resampled every 'step' m and lightly smoothed."""
    from scipy.ndimage import gaussian_filter1d
    P = np.asarray(xyz, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-3]; P, d = P[keep], d[keep]
    s = np.arange(0, d[-1], step)
    P = np.stack([np.interp(s, d, P[:, k]) for k in range(3)], 1)
    P = gaussian_filter1d(P, 2, axis=0, mode='nearest')
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    F = np.gradient(P, axis=0); F[:, 2] = 0; F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return P, d, F


def axis_path(V, ground_at=None):
    """No poses / drawn path: the data's main horizontal axis, following the middle of the data (2 m bins)."""
    from scipy.ndimage import gaussian_filter1d
    xy = V[:, :2]; c = xy.mean(0)
    _, _, Wt = np.linalg.svd(xy[::max(1, len(xy) // 200000)] - c, full_matrices=False); u = Wt[0]; w = np.array([-u[1], u[0]])
    a = (xy - c) @ u; l = (xy - c) @ w
    bins = np.arange(np.percentile(a, 0.5), np.percentile(a, 99.5) + 2, 2.0)
    mid = np.array([np.median(l[(a >= b) & (a < b + 2)]) if ((a >= b) & (a < b + 2)).sum() > 10 else np.nan for b in bins])
    ok = np.isfinite(mid); mid = np.interp(bins, bins[ok], mid[ok]); mid = gaussian_filter1d(mid, 3, mode='nearest')
    P = c + bins[:, None] * u + mid[:, None] * w
    z = ground_at(P) if ground_at else np.full(len(P), np.percentile(V[:, 2], 10))
    return path_from_points(np.c_[P, z])


def save_path(project, P, arc, F, source):
    np.savez(os.path.join(project.mesh_dir, 'path.npz'), P=P, arc=arc, F=F)
    project.data['path'] = dict(source=source, length=float(arc[-1]), n=int(len(P))); project.save()


# ------------------------------------------------------------------------------------------- ground
def ground_grid(points, cell, pad=20.0):
    """Ground height per cell from any surface points (mesh vertices, LiDAR, splat centres): a low percentile per
    cell, then cells sticking far above their neighbourhood's lowest ground (canopy, roofs, cars) are dropped and
    filled from the ground around them."""
    from scipy.ndimage import gaussian_filter, minimum_filter
    lo = points[:, :2].min(0) - pad; hi = points[:, :2].max(0) + pad
    nx, ny = (np.ceil((hi - lo) / cell)).astype(int) + 1
    gi = ((points[:, 0] - lo[0]) / cell).astype(int); gj = ((points[:, 1] - lo[1]) / cell).astype(int)
    cid = gi * ny + gj; z = points[:, 2]
    order = np.lexsort((z, cid)); cs = cid[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); counts = np.diff(np.r_[starts, len(cs)])
    g = np.full(nx * ny, np.nan, np.float32)
    good = counts >= 2
    g[cs[starts[good]]] = z[order[starts[good] + (0.1 * counts[good]).astype(int)]]
    g = g.reshape(nx, ny)
    # progressive check against the local lowest ground (windows of 2.5 m and 10 m)
    for w, tol in ((5, 0.6), (21, 1.5)):
        mn = minimum_filter(np.where(np.isfinite(g), g, np.inf), size=w)
        g = np.where(np.isfinite(g) & (g - mn < tol + 0.05 * w * cell * 4), g, np.nan)
    filled = core.fill_grid(g)
    return gaussian_filter(filled, 1.0).astype(np.float32), lo, float(np.isfinite(g).mean())


def ground_sampler(gz, lo, cell):
    nx, ny = gz.shape

    def at(xy):
        fx = (xy[:, 0] - lo[0]) / cell; fy = (xy[:, 1] - lo[1]) / cell
        i0 = np.clip(fx.astype(int), 0, nx - 2); j0 = np.clip(fy.astype(int), 0, ny - 2)
        ax = np.clip(fx - i0, 0, 1); ay = np.clip(fy - j0, 0, 1)
        return (gz[i0, j0] * (1 - ax) * (1 - ay) + gz[i0 + 1, j0] * ax * (1 - ay) + gz[i0, j0 + 1] * (1 - ax) * ay + gz[i0 + 1, j0 + 1] * ax * ay)
    return at


# ------------------------------------------------------------------------------------------- mesh chunks
def prep_mesh(project, meshes, ground_at, route):
    """Meshes [(name, V, C, T)] -> chunks of CHUNK m along the path x CHUNK m across it."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    import shutil
    t0 = time.time()
    final = os.path.join(project.mesh_dir, 'chunks')
    cdir = os.path.join(project.mesh_dir, f'chunks_new_{int(time.time())}')     # built aside, swapped in at the end
    O = route.origin; CH = core.CHUNK
    chunks, comp_base, parts = [], 0, []
    for pi, (name, V, C, Tr) in enumerate(meshes):
        parts.append(name); nv = len(V)
        A = coo_matrix((np.ones(2 * len(Tr)), (np.r_[Tr[:, 0], Tr[:, 1]], np.r_[Tr[:, 1], Tr[:, 2]])), shape=(nv, nv))
        ncomp, lab = connected_components(A, directed=False)
        tcomp = lab[Tr[:, 0]]
        csize = np.bincount(tcomp, minlength=ncomp)[tcomp].astype(np.int32)
        cen = V[Tr].mean(1)
        arc, lat = route.project(cen[:, :2])
        H = cen[:, 2] - ground_at(cen[:, :2])
        b = np.clip((arc // CH).astype(int), -100, int(route.arc[-1] // CH) + 100)
        lb = np.floor(lat / CH + 0.5).astype(int)
        key2 = (b + 1000) * 10000 + (lb + 5000)
        for kk in np.unique(key2):
            sel = np.flatnonzero(key2 == kk); bin_, lbin = kk // 10000 - 1000, kk % 10000 - 5000
            used, inv = np.unique(Tr[sel].ravel(), return_inverse=True)
            key = f'p{pi + 1:02d}_b{bin_:03d}' + ((f'_l{lbin}' if lbin > 0 else f'_lm{-lbin}') if lbin else '')
            d = os.path.join(cdir, key); os.makedirs(d, exist_ok=True)
            Vc = (V[used] - O).astype(np.float32)
            for n_, arr in (('V', Vc), ('C', C[used]), ('T', inv.reshape(-1, 3).astype(np.uint32)),
                            ('H', np.clip(H[sel] * 100, -32000, 32000).astype(np.int16)), ('comp', (tcomp[sel] + comp_base).astype(np.int64)),
                            ('csize', csize[sel]), ('arc', arc[sel].astype(np.float32)), ('lat', lat[sel].astype(np.float32))):
                np.save(os.path.join(d, n_ + '.npy'), arr)
            chunks.append(dict(key=key, part=pi + 1, part_file=name, bin=int(bin_), lbin=int(lbin),
                               arc0=float(arc[sel].min()), arc1=float(arc[sel].max()),
                               bbox=np.r_[Vc.min(0), Vc.max(0)].round(2).tolist(), nv=int(len(used)), nt=int(len(sel))))
        comp_base += ncomp
        log(f'  {name}: {len(Tr):,} triangles, {ncomp:,} pieces ({time.time() - t0:.0f} s)')
    if os.path.exists(final):
        old = os.path.join(project.mesh_dir, f'chunks_old_{int(time.time())}'); os.rename(final, old)
    os.rename(cdir, final)
    for d in os.listdir(project.mesh_dir):
        if d.startswith('chunks_old_'):
            shutil.rmtree(os.path.join(project.mesh_dir, d), ignore_errors=True)
    json.dump(dict(origin=O.tolist(), chunk=CH, chunks=chunks, parts=parts), open(os.path.join(project.mesh_dir, 'meta.json'), 'w'))
    log(f'mesh chunks: {len(chunks)} ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------- orthophoto
def prep_ortho(project, G):
    """Top-down tiles from ground-band coloured samples G = [x, y, z, r, g, b, w] (z-up, colours 0..1)."""
    import cv2
    t0 = time.time(); TILES = project.tiles_dir; PX0, T, LEVELS, PAD = core.PX0, core.T, core.LEVELS, core.PAD
    for f in os.listdir(TILES):
        p = os.path.join(TILES, f)
        if os.path.isdir(p):
            import shutil; shutil.rmtree(p, ignore_errors=True)
    np.save(os.path.join(TILES, 'surface_xyz.npy'), G[:, 0:3].astype(np.float32))
    X0 = math.floor(G[:, 0].min()) - 1.0; YTOP = math.ceil(G[:, 1].max()) + 1.0
    tw = T * PX0
    ti = ((G[:, 0] - X0) / tw).astype(int); tj = ((YTOP - G[:, 1]) / tw).astype(int)
    key = ti * 100000 + tj
    order = np.argsort(key); G, key = G[order], key[order]
    ukeys, ustart = np.unique(key, return_index=True)
    span = {int(k): (int(s), int(e)) for k, s, e in zip(ukeys, ustart, np.r_[ustart[1:], len(key)])}
    tiles0 = {}
    for k in sorted(span):
        i, j = k // 100000, k % 100000
        g = np.concatenate([G[span[kk][0]:span[kk][1]] for kk in
                            [(i + di) * 100000 + (j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)] if kk in span])
        u = ((g[:, 0] - X0) / PX0 - i * T + PAD).astype(int); v = ((YTOP - g[:, 1]) / PX0 - j * T + PAD).astype(int)
        S = T + 2 * PAD
        ok = (u >= 0) & (u < S) & (v >= 0) & (v < S); u, v, g = u[ok], v[ok], g[ok]
        lin = v * S + u
        Wt = np.bincount(lin, g[:, 6], S * S).reshape(S, S)
        C = np.stack([np.bincount(lin, g[:, 3 + c] * g[:, 6], S * S).reshape(S, S) for c in range(3)], -1)
        img, sup = core.push_pull(C, Wt)
        sup = cv2.resize((sup > 0).astype(np.float32), (S, S), interpolation=cv2.INTER_LINEAR)
        bg = np.array([0.15, 0.157, 0.172])
        img = (np.clip(img * sup[..., None] + bg * (1 - sup[..., None]), 0, 1) * 255).astype(np.uint8)
        tiles0[(i, j)] = img[PAD:PAD + T, PAD:PAD + T, ::-1].copy()
    exist, cur = {}, tiles0
    for L in range(LEVELS):
        d = os.path.join(TILES, str(L)); os.makedirs(d, exist_ok=True)
        for (i, j), img in cur.items():
            cv2.imwrite(os.path.join(d, f'{i}_{j}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        exist[L] = sorted(cur)
        if L == LEVELS - 1:
            break
        nxt = {}
        for (i, j) in {(i // 2, j // 2) for (i, j) in cur}:
            big = np.full((2 * T, 2 * T, 3), (44, 40, 38), np.uint8)
            for di in (0, 1):
                for dj in (0, 1):
                    c = cur.get((2 * i + di, 2 * j + dj))
                    if c is not None:
                        big[dj * T:(dj + 1) * T, di * T:(di + 1) * T] = c
            nxt[(i, j)] = cv2.resize(big, (T, T), interpolation=cv2.INTER_AREA)
        cur = nxt
    json.dump(dict(px0=PX0, tile=T, levels=LEVELS, x0=X0, ztop=YTOP,
                   tiles={str(L): [f'{i}_{j}' for i, j in v] for L, v in exist.items()}),
              open(os.path.join(TILES, 'meta.json'), 'w'))
    log(f'orthophoto: {len(tiles0)} tiles ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------- splat tiles
def prep_splats(project, ground_at=None):
    """Each splat layer -> spatial tiles for streaming: per gaussian centre (3 f32), covariance (6 f32), rgba (4 u8)."""
    out = []
    for L in project.layers:
        if L['kind'] != 'splat':
            continue
        t0 = time.time()
        d = project.dir('splat', L['id'])
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
        X, rgb, op, sc, q = read_splat(L['src'], project.T(L))
        keep = op > 0.02
        X, rgb, op, sc, q = X[keep], rgb[keep], op[keep], sc[keep], q[keep]
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        S = np.exp(np.clip(sc, -12, 3))
        M = R * S[:, None, :]
        Cov = M @ M.transpose(0, 2, 1)
        cov6 = np.c_[Cov[:, 0, 0], Cov[:, 0, 1], Cov[:, 0, 2], Cov[:, 1, 1], Cov[:, 1, 2], Cov[:, 2, 2]].astype(np.float32)
        rgba = np.c_[rgb * 255, op * 255].round().clip(0, 255).astype(np.uint8)
        TS = 25.0
        ti = np.floor(X[:, 0] / TS).astype(int); tj = np.floor(X[:, 1] / TS).astype(int)
        k = ti * 100000 + tj; order = np.argsort(k, kind='stable'); k = k[order]
        u, st = np.unique(k, return_index=True); en = np.r_[st[1:], len(k)]
        tiles = []
        for kk, a, b in zip(u, st, en):
            idx = order[a:b]
            idx = idx[np.argsort(-(rgba[idx, 3].astype(np.float32) * np.exp(sc[idx].max(1))))]   # most visible first (LOD = prefix)
            name = f'{kk // 100000}_{kk % 100000}'
            with open(os.path.join(d, name + '.bin'), 'wb') as f:
                f.write(X[idx].astype(np.float32).tobytes()); f.write(cov6[idx].tobytes()); f.write(rgba[idx].tobytes())
            tiles.append(dict(name=name, n=int(len(idx)), bbox=np.r_[X[idx].min(0), X[idx].max(0)].round(2).tolist()))
        json.dump(dict(id=L['id'], tile=TS, tiles=tiles, n=int(len(X))), open(os.path.join(d, 'meta.json'), 'w'))
        log(f'splat {L["name"]}: {len(X):,} gaussians in {len(tiles)} tiles ({time.time() - t0:.0f} s)')
        out.append(L['id'])
    return out


# ------------------------------------------------------------------------------------------- all
def prepare(project, ortho_source='auto'):
    core.configure(project)
    S = project.settings
    meshes = []
    for L in project.layers:
        if L['kind'] == 'mesh':
            for f in project.files_of(L):
                V, C, Tr = read_mesh(f, project.T(L))
                meshes.append((os.path.basename(f), V, C, Tr))
    pts = [read_points(f, project.T(L)) for L in project.layers if L['kind'] == 'points' for f in project.files_of(L)]
    splat_pts = []
    for L in project.layers:
        if L['kind'] == 'splat' and L.get('visible', True):       # orthophoto / ground from the visible splats
            X, rgb, op, sc, q = read_splat(L['src'], project.T(L), max_n=6_000_000)
            k = (op > 0.3) & (sc.max(1) < np.log(0.3)); splat_pts.append((X[k], rgb[k], op[k]))
    allV = np.concatenate([m[1] for m in meshes] + [p[0] for p in pts] + [s[0] for s in splat_pts])
    log(f'layers: {len(meshes)} mesh files, {len(pts)} point files, {len(splat_pts)} splats; {len(allV):,} surface samples')
    gz, glo, frac = ground_grid(allV, float(S['ground_cell']))
    ground_at = ground_sampler(gz, glo, float(S['ground_cell']))
    log(f'ground grid {gz.shape[0]} x {gz.shape[1]} ({frac:.0%} measured)')
    # path: keep a path from poses / drawn; else the data's main axis
    if not os.path.exists(os.path.join(project.mesh_dir, 'path.npz')):
        P, arc, F = axis_path(allV, ground_at); save_path(project, P, arc, F, 'data axis')
        log(f'path: data main axis, {arc[-1]:.0f} m')
    route = core.Route()
    np.savez(os.path.join(project.mesh_dir, 'ground_grid.npz'), z=gz, x0=glo[0], y0=glo[1], step=float(S['ground_cell']), origin=route.origin)
    # mesh chunks: rebuilt only when the mesh inputs / chunking / path changed (edits are stored per chunk triangle)
    sig = json.dumps(dict(files=[(f, os.path.getmtime(f), os.path.getsize(f)) for L in project.layers if L['kind'] == 'mesh' for f in project.files_of(L)],
                          T=[L['transform'] for L in project.layers if L['kind'] == 'mesh'], chunk=core.CHUNK,
                          path=os.path.getmtime(os.path.join(project.mesh_dir, 'path.npz'))))
    sig_f = os.path.join(project.mesh_dir, 'inputs.json')
    same = os.path.exists(sig_f) and open(sig_f).read() == sig and os.path.exists(os.path.join(project.mesh_dir, 'meta.json'))
    if meshes and not same:
        ed = project.edits_dir
        if any(os.scandir(os.path.join(ed, 'ops'))) if os.path.isdir(os.path.join(ed, 'ops')) else False:
            bk = ed.rstrip('/') + time.strftime('_backup_%Y%m%d_%H%M%S'); os.rename(ed, bk)
            log(f'mesh inputs changed: previous edits kept in {bk} (they belong to the old triangles)')
        prep_mesh(project, meshes, ground_at, route)
        open(sig_f, 'w').write(sig)
    elif meshes:
        log('mesh unchanged: chunks and edits kept')
    # orthophoto: splat > points > mesh colours, ground band only (no canopy / roofs)
    srcs = splat_pts if splat_pts and ortho_source in ('auto', 'splat') else \
        [(p[0], p[1] / 255.0, np.ones(len(p[0]))) for p in pts] if pts and ortho_source in ('auto', 'points') else \
        [(m[1], m[2] / 255.0, np.ones(len(m[1]))) for m in meshes]
    G = np.concatenate([np.c_[X, c, w] for X, c, w in srcs]).astype(np.float32)
    rel = G[:, 2] - ground_at(G[:, :2])
    prep_ortho(project, G[(rel > -0.35) & (rel < 0.5)])
    prep_splats(project)
    project.data['prepared'] = time.strftime('%Y-%m-%d %H:%M:%S'); project.save()
    log('ready')
