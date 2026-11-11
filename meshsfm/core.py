"""MeshSfM SplatStudio core: chunked mesh store with non-destructive edits (masks, vertex edits, generated patches,
undo / redo), road tools, orthophoto colour lookup, lane-line export. All coordinates are the project's internal
frame (z up, metres); see project.py."""
import glob, json, math, os, shutil, threading, time

import numpy as np

from .project import apply, apply_dir

PROJECT = None
TILES = ANN = MESH = EDITS = PARTS_OUT = ROUTE_OUT = None
PX0, T, LEVELS, PAD = 0.025, 1024, 6, 48   # orthophoto: pixel (m), tile size (px), pyramid levels, padding
CHUNK = 50.0
GRID = 0.5


def configure(project):
    """Point the module at a project (folders, settings)."""
    global PROJECT, TILES, ANN, MESH, EDITS, PARTS_OUT, ROUTE_OUT, PX0, CHUNK, GRID
    PROJECT = project
    TILES, ANN, MESH, EDITS = project.tiles_dir, project.ann_dir, project.mesh_dir, project.edits_dir
    name = project.data.get('name', 'mesh')
    PARTS_OUT = os.path.join(project.export_dir, 'parts')
    ROUTE_OUT = os.path.join(project.export_dir, name)
    PX0, CHUNK, GRID = float(project.settings['ortho_px']), float(project.settings['chunk']), float(project.settings['ground_cell'])


def to_source(V):
    """internal z-up -> frame of the first mesh layer (for exports)."""
    return apply(PROJECT.source_frame()[0], V)


def to_source_dir(N):
    return apply_dir(PROJECT.source_frame()[0], N)


# =========================================================================================== utils
def load_ply_memmap(path):
    with open(path, 'rb') as f:
        n, props = None, []
        while True:
            line = f.readline().decode().strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        off = f.tell()
    return np.memmap(path, dtype='<f4', mode='r', offset=off, shape=(n, len(props))), props


def fill_grid(G, levels=12):
    """Fill NaNs of a 2D grid smoothly from coarser averages (push-pull)."""
    import cv2
    V = np.nan_to_num(G).astype(np.float32); W = np.isfinite(G).astype(np.float32)
    Vs, Ws = [V * W], [W]
    for _ in range(levels):
        v, w = Vs[-1], Ws[-1]
        if min(v.shape) < 2:
            break
        h, wd = (v.shape[0] // 2) * 2, (v.shape[1] // 2) * 2
        v, w = v[:h, :wd], w[:h, :wd]
        Vs.append(v[0::2, 0::2] + v[1::2, 0::2] + v[0::2, 1::2] + v[1::2, 1::2])
        Ws.append(w[0::2, 0::2] + w[1::2, 0::2] + w[0::2, 1::2] + w[1::2, 1::2])
    out = Vs[-1] / np.maximum(Ws[-1], 1e-6)
    if not (Ws[-1] > 0).all():
        out[Ws[-1] == 0] = np.nanmean(G)
    for k in range(len(Vs) - 2, -1, -1):
        up = cv2.resize(out, (Vs[k].shape[1], Vs[k].shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(Ws[k], 0, 1)
        out = a * (Vs[k] / np.maximum(Ws[k], 1e-6)) + (1 - a) * up
    res = np.full(G.shape, np.nan, np.float32)
    res[:out.shape[0], :out.shape[1]] = out
    if np.isnan(res).any():  # odd edge rows/cols
        from scipy.ndimage import distance_transform_edt
        _, (ii, jj) = distance_transform_edt(np.isnan(res), return_indices=True)
        res = res[ii, jj]
    return res


class Route:
    """Drive path (z-up): positions, arc, forward; projection of points to (arc, lateral)."""
    def __init__(self):
        p = np.load(os.path.join(MESH, 'path.npz'))
        self.P = p['P'].astype(np.float64)
        self.arc = p['arc'].astype(np.float64)
        f = p['F'][:, :2].astype(np.float64)
        self.F = f / np.linalg.norm(f, axis=1, keepdims=True)
        try:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(self.P[:, :2])
        except ImportError:
            self.tree = None
        self.origin = np.round(self.P.mean(0), 0)

    def project(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        if self.tree is not None:
            _, j = self.tree.query(xy)
        else:
            P2 = self.P[:, :2]
            j = np.empty(len(xy), dtype=np.int64)
            for b in range(0, len(xy), 4096):
                 sub = xy[b:b + 4096]
                 j[b:b + 4096] = ((sub[:, None, :] - P2[None, :, :]) ** 2).sum(-1).argmin(1)
        rel = xy - self.P[j, :2]
        f = self.F[j]
        return self.arc[j] + (rel * f).sum(1), rel[:, 0] * f[:, 1] - rel[:, 1] * f[:, 0]  # arc, lat (+ right)

    def at(self, arc):
        return np.stack([np.interp(arc, self.arc, self.P[:, k]) for k in range(3)], -1)

    def ground_cap(self, xy):
        """Highest plausible ground (z-up) near the road: the road height + a little, rising slowly with distance
        from the centre; +inf beyond 6 m. Used so tree canopy over the road is never taken for the ground."""
        arc, lat = self.project(xy)
        zr = np.interp(arc, self.arc, self.P[:, 2])
        cap = zr + 0.25 + 0.1 * np.maximum(0, np.abs(lat) - 2.5)
        return np.where(np.abs(lat) < 6.0, cap, np.inf)


# ======================================================================================= ortho prep
def push_pull(C, W, thr=0.6, levels=8):
    import cv2
    Cs, Ws = [C], [W]
    for _ in range(levels):
        c, w = Cs[-1], Ws[-1]
        h, wd = (c.shape[0] // 2) * 2, (c.shape[1] // 2) * 2
        c, w = c[:h, :wd], w[:h, :wd]
        Cs.append(c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2])
        Ws.append(w[0::2, 0::2] + w[1::2, 0::2] + w[0::2, 1::2] + w[1::2, 1::2])
    out = Cs[-1] / np.maximum(Ws[-1], 1e-6)[..., None]
    for k in range(levels - 1, -1, -1):
        c, w = Cs[k], Ws[k]
        up = cv2.resize(out, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(w / thr, 0, 1)[..., None]
        out = a * (c / np.maximum(w, 1e-6)[..., None]) + (1 - a) * up
    return out, Ws[6]


class Ortho:
    """Colour lookup in the level-0 orthophoto tiles (z-up world xy)."""
    def __init__(self):
        self.meta = json.load(open(os.path.join(TILES, 'meta.json')))
        self.have = set(self.meta['tiles']['0'])
        self.cache = {}

    def colours(self, xy):
        import cv2
        m = self.meta
        u = (xy[:, 0] - m['x0']) / PX0; v = (m['ztop'] - xy[:, 1]) / PX0
        ti, tj = (u // T).astype(int), (v // T).astype(int)
        out = np.full((len(xy), 3), 110, np.uint8)
        for i, j in set(zip(ti.tolist(), tj.tolist())):
            name = f'{i}_{j}'
            if name not in self.have:
                continue
            if name not in self.cache:
                if len(self.cache) > 64:
                    self.cache.clear()
                self.cache[name] = cv2.imread(os.path.join(TILES, '0', name + '.jpg'))[:, :, ::-1]
            k = (ti == i) & (tj == j)
            out[k] = self.cache[name][np.clip((v[k] - j * T).astype(int), 0, T - 1), np.clip((u[k] - i * T).astype(int), 0, T - 1)]
        return out

    def area(self, xy, cell=0.05):
        """Colour re-projected from the splat orthophoto, averaged over a vertex's cell (4 sub-samples), so paint stays sharp."""
        o = max(cell, PX0) * 0.25; acc = np.zeros((len(xy), 3))
        for dx, dy in ((-o, -o), (-o, o), (o, -o), (o, o)):
            acc += self.colours(xy + np.array([dx, dy]))
        return np.round(acc / 4).astype(np.uint8)


# ======================================================================================= mesh store
def grid_patch(xy0, cell, mask, Z, Cc, O):
    """Triangles over grid cells (vertices at cell centres) where all four corners are in mask."""
    nx, ny = mask.shape
    idx = -np.ones((nx, ny), np.int64)
    idx[mask] = np.arange(mask.sum())
    ii, jj = np.nonzero(mask)
    V = np.c_[xy0[0] + (ii + 0.5) * cell, xy0[1] + (jj + 0.5) * cell, Z[mask]] - O
    q = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    a, b, c, d = idx[:-1, :-1][q], idx[1:, :-1][q], idx[:-1, 1:][q], idx[1:, 1:][q]
    Tt = np.r_[np.c_[a, b, d], np.c_[a, d, c]]
    return V.astype(np.float32), Cc[mask].astype(np.uint8), Tt.astype(np.uint32)


def laplace_fill(known, val, solve_mask):
    """Harmonic interpolation: cells in solve_mask get values with zero Laplacian; 'known' cells fixed."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sl
    nx, ny = solve_mask.shape
    unk = solve_mask & ~known
    n = int(unk.sum())
    out = np.where(known[..., None] if val.ndim == 3 else known, val, 0.0).astype(np.float64)
    if n == 0:
        return out
    id_ = -np.ones((nx, ny), np.int64); id_[unk] = np.arange(n)
    ii, jj = np.nonzero(unk)
    rows, cols, vals = [], [], []
    rhs = np.zeros((n,) + val.shape[2:])
    deg = np.zeros(n)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        i2, j2 = ii + di, jj + dj
        ok = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny)
        i2c, j2c = np.clip(i2, 0, nx - 1), np.clip(j2, 0, ny - 1)
        nb_unk = ok & unk[i2c, j2c]; nb_known = ok & known[i2c, j2c]
        deg += nb_unk | nb_known
        r = np.flatnonzero(nb_unk)
        rows.append(r); cols.append(id_[i2c[r], j2c[r]]); vals.append(-np.ones(len(r)))
        r = np.flatnonzero(nb_known)
        np.add.at(rhs, r, val[i2c[r], j2c[r]])
    deg = np.maximum(deg, 1)
    A = sp.csr_matrix((np.r_[deg, np.concatenate(vals)], (np.r_[np.arange(n), np.concatenate(rows)],
                                                         np.r_[np.arange(n), np.concatenate(cols)])), shape=(n, n))
    sol = sl.spsolve(A.tocsc(), rhs)
    out[unk] = sol if sol.ndim == rhs.ndim else sol.reshape(rhs.shape)
    return out


class MeshStore:
    def __init__(self, route):
        self.route = route
        self.meta = json.load(open(os.path.join(MESH, 'meta.json')))
        self.O = np.array(self.meta['origin'])
        self.chunks = {c['key']: c for c in self.meta['chunks']}
        g = np.load(os.path.join(MESH, 'ground_grid.npz'))
        self.gz, self.gx0, self.gy0, self.gstep = g['z'], float(g['x0']), float(g['y0']), float(g['step'])
        self.lock = threading.RLock()
        self.cache = {}
        self.deleted = {}
        self.version = 0
        self.changes = []      # (version, keys)
        self.patches = {}      # pid -> dict(kind, key, V, C, T, active, label)
        self.ops = []          # applied ops (undo stack)
        self.redo = []
        self.proposal = None
        self.status = dict(busy=False, msg='', progress=0.0, result=None)
        self.ortho = Ortho()
        self.lod = {}
        os.makedirs(os.path.join(EDITS, 'del'), exist_ok=True)
        os.makedirs(os.path.join(EDITS, 'patches'), exist_ok=True)
        os.makedirs(os.path.join(EDITS, 'ops'), exist_ok=True)
        os.makedirs(os.path.join(EDITS, 'vert'), exist_ok=True)
        for f in glob.glob(os.path.join(EDITS, 'del', '*.npy')):
            k = os.path.basename(f)[:-4]
            if k in self.chunks:
                self.deleted[k] = np.load(f)
        # vertex edits (smoothing, levelled fills): per chunk z (NaN = original) and colour (-1 = original)
        self.vz, self.vc, self.touched, self._hcache, self._alive = {}, {}, {}, {}, {}
        for f in glob.glob(os.path.join(EDITS, 'vert', '*.npz')):
            k = os.path.basename(f)[:-4]
            if k in self.chunks:
                z = np.load(f)
                self._varrays(k)
                self.vz[k][z['idx']] = z['z']; self.vc[k][z['idx']] = z['c']
                self.touched[k][z['idx']] = True
        st = os.path.join(EDITS, 'state.json')
        if os.path.exists(st):
            s = json.load(open(st))
            for pid, info in s.get('patches', {}).items():
                z = np.load(os.path.join(EDITS, 'patches', pid + '.npz'))
                self.patches[pid] = dict(info, V=z['V'], C=z['C'], T=z['T'])
            self.ops = s.get('ops', []); self.redo = s.get('redo', [])
        self.corr = None

    # ---------------------------------------------------------------- data access
    def load(self, key):
        with self.lock:
            if key not in self.cache:
                if len(self.cache) > 40:
                    self.cache.pop(next(iter(self.cache)))
                d = os.path.join(MESH, 'chunks', key)
                self.cache[key] = {n: np.load(os.path.join(d, n + '.npy'), mmap_mode='r')
                                   for n in ('V', 'C', 'T', 'H', 'comp', 'csize', 'arc', 'lat')}
            return self.cache[key]

    def dmask(self, key):
        if key not in self.deleted:
            self.deleted[key] = np.zeros(self.chunks[key]['nt'], bool)
        return self.deleted[key]

    def ground_at(self, xy):
        nx, ny = self.gz.shape
        fx = (xy[:, 0] - self.gx0) / self.gstep; fy = (xy[:, 1] - self.gy0) / self.gstep
        i0 = np.clip(fx.astype(int), 0, nx - 2); j0 = np.clip(fy.astype(int), 0, ny - 2)
        ax = np.clip(fx - i0, 0, 1); ay = np.clip(fy - j0, 0, 1)
        g = self.gz
        return (g[i0, j0] * (1 - ax) * (1 - ay) + g[i0 + 1, j0] * ax * (1 - ay) + g[i0, j0 + 1] * (1 - ax) * ay + g[i0 + 1, j0 + 1] * ax * ay)

    def keys_in(self, arc0, arc1):
        return [k for k, c in self.chunks.items() if c['arc1'] >= arc0 and c['arc0'] <= arc1]

    def keys_near(self, xy0, xy1, pad=2.0):
        out = []
        for k, c in self.chunks.items():
            b = np.array(c['bbox'])
            if b[3] + self.O[0] >= xy0[0] - pad and b[0] + self.O[0] <= xy1[0] + pad and b[4] + self.O[1] >= xy0[1] - pad and b[1] + self.O[1] <= xy1[1] + pad:
                out.append(k)
        return out

    def centroids(self, key):
        d = self.load(key)
        return self.verts(key)[0][np.asarray(d['T'])].mean(1) + self.O

    # ---------------------------------------------------------------- vertex edits
    def _varrays(self, key):
        if key not in self.vz:
            nv = self.chunks[key]['nv']
            self.vz[key] = np.full(nv, np.nan, np.float32)
            self.vc[key] = np.full((nv, 3), -1, np.int16)
            self.touched[key] = np.zeros(nv, bool)

    def _prop_verts(self, key):
        p = self.proposal
        return p and p.get('verts', {}).get(key)

    def verts(self, key, preview=False, idx=None):
        """Current vertex positions (rel O) and colours, with vertex edits (and the pending proposal if preview)."""
        d = self.load(key)
        V = np.array(d['V'] if idx is None else np.asarray(d['V'])[idx])
        C = np.array(d['C'] if idx is None else np.asarray(d['C'])[idx])
        if key in self.vz:
            z, c = self.vz[key], self.vc[key]
            if idx is not None:
                z, c = z[idx], c[idx]
            m = np.isfinite(z); V[m, 2] = z[m]
            m = c[:, 0] >= 0; C[m] = c[m]
        pv = self._prop_verts(key) if preview else None
        if pv:
            pi, pz, pc = pv
            if idx is None:
                V[pi, 2] = pz; C[pi] = pc
            else:
                pos = np.searchsorted(pi, idx); pos = np.clip(pos, 0, len(pi) - 1); hit = pi[pos] == idx
                V[hit, 2] = pz[pos[hit]]; C[hit] = pc[pos[hit]]
        return V, C

    def alive(self, key):
        """Vertices used by at least one kept triangle."""
        ver = self.chunks[key].get('ver', 0)
        if key not in self._alive or self._alive[key][0] != ver:
            d = self.load(key)
            a = np.zeros(self.chunks[key]['nv'], bool)
            a[np.asarray(d['T'])[~self.dmask(key)].ravel()] = True
            self._alive[key] = (ver, a)
        return self._alive[key][1]

    def Hcur(self, key):
        """Triangle height above the ground grid (cm, int16), recomputed where vertices were edited."""
        d = self.load(key)
        if key not in self.vz or not np.isfinite(self.vz[key]).any():
            return np.asarray(d['H'])
        ver = self.chunks[key].get('vver', 0)
        if key not in self._hcache or self._hcache[key][0] != ver:
            cen = self.centroids(key)
            self._hcache[key] = (ver, np.clip(np.round((cen[:, 2] - self.ground_at(cen[:, :2])) * 100), -32000, 32000).astype(np.int16))
        return self._hcache[key][1]

    def _set_verts(self, key, idx, z, c):
        self._varrays(key)
        self.vz[key][idx] = z; self.vc[key][idx] = c; self.touched[key][idx] = True
        self.chunks[key]['vver'] = self.chunks[key].get('vver', 0) + 1

    def gather(self, bb0, bb1, alive=True):
        """Edited vertices with xy inside a box: [(key, idx, V abs, C)]."""
        out = []
        for k in self.keys_near(bb0, bb1):
            V0 = np.asarray(self.load(k)['V'])
            x, y = V0[:, 0] + self.O[0], V0[:, 1] + self.O[1]
            m = (x >= bb0[0]) & (x <= bb1[0]) & (y >= bb0[1]) & (y <= bb1[1])
            if alive:
                m &= self.alive(k)
            idx = np.flatnonzero(m)
            if len(idx):
                V, C = self.verts(k, idx=idx)
                out.append((k, idx, V.astype(np.float64) + self.O, C.astype(np.float64)))
        return out

    def vdelta_bytes(self, key):
        """All vertices edited in this session (current values, with the preview): the client patches its buffers."""
        t = self.touched.get(key)
        idx = np.flatnonzero(t) if t is not None else np.zeros(0, np.int64)
        V, C = self.verts(key, preview=True)
        return self.pack(dict(key=key, n=int(len(idx))), [idx.astype(np.uint32), V[idx].astype(np.float32), C[idx].astype(np.uint8)])

    def coverage(self, bb0, cell, nx, ny, override=None, skip=None, zref=None, band=0.6):
        """Which grid cells are covered by kept triangles (xy footprint), with the lowest surface z and the
        colour there. 'override' = proposed vertex edits {key: (idx, z, c)}, 'skip' = {key: tri idx} to ignore.
        zref = expected surface height per cell: only triangles within 'band' of it count (canopy is not ground)."""
        cov = np.zeros((nx, ny), bool); zmin = np.full((nx, ny), np.inf); col = np.zeros((nx, ny, 3)); cnt = np.zeros((nx, ny))
        bb1 = bb0 + np.array([nx, ny]) * cell
        for k in self.keys_near(bb0, bb1, pad=1.0):
            d = self.load(k)
            V, C = self.verts(k)
            if override and k in override:
                pi, pz, pc = override[k]; V[pi, 2] = pz; C[pi] = pc
            Tt = np.asarray(d['T'])
            keep = ~self.dmask(k)
            if skip and k in skip:
                keep = keep.copy(); keep[skip[k]] = False
            ce = V[Tt[:, 0], :2] + self.O[:2]
            keep &= (ce[:, 0] > bb0[0] - 1) & (ce[:, 0] < bb1[0] + 1) & (ce[:, 1] > bb0[1] - 1) & (ce[:, 1] < bb1[1] + 1)
            tri = Tt[keep].astype(np.int64)
            if not len(tri):
                continue
            P = V[tri].astype(np.float64) + self.O           # (n, 3, 3)
            Q = C[tri].astype(np.float64)
            L = np.max(np.linalg.norm(P[:, [1, 2, 0], :2] - P[:, :, :2], axis=2), axis=1)
            sub = np.clip(np.ceil(L / (0.7 * cell)), 1, 24).astype(int)
            for s in np.unique(sub):
                g = np.flatnonzero(sub == s)
                a, b = np.meshgrid(np.arange(s + 1), np.arange(s + 1), indexing='ij'); m = a + b <= s
                B = np.c_[a[m], b[m], s - a[m] - b[m]] / s                      # barycentric lattice
                pts = np.einsum('kj,njd->nkd', B, P[g]).reshape(-1, 3)
                cs = np.einsum('kj,njd->nkd', B, Q[g]).reshape(-1, 3)
                ii = ((pts[:, 0] - bb0[0]) / cell).astype(int); jj = ((pts[:, 1] - bb0[1]) / cell).astype(int)
                ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                if zref is not None:
                    ok[ok] &= np.abs(pts[ok, 2] - zref[ii[ok], jj[ok]]) < band
                ii, jj = ii[ok], jj[ok]
                cov[ii, jj] = True
                np.minimum.at(zmin, (ii, jj), pts[ok, 2])
                np.add.at(col, (ii, jj), cs[ok]); np.add.at(cnt, (ii, jj), 1)
        col /= np.maximum(cnt, 1)[..., None]
        return cov, zmin, col

    def under_patch(self, bb0, cell, nx, ny, region, override=None, skip=None, below=0.015, full=False, zref=None):
        """Close holes inside 'region' with a grid surface just under the mesh (below the lowest mesh point of
        the neighbouring cells), heights and colours blended from the mesh. full=True lays it under the whole
        region (+2 cells): any sliver / crack there shows ground, never sky. Otherwise only around holes."""
        from scipy.ndimage import minimum_filter
        cov, zmin, col = self.coverage(bb0, cell, nx, ny, override, skip, zref)
        hole = region & ~cov
        if not full and hole.sum() < 1:
            return None
        mask = self._dilate(region, 2) if full else self._dilate(hole, 3) & self._dilate(region, 2)
        known = cov & mask
        if known.sum() < 3:
            return None
        zlow = minimum_filter(np.where(cov, zmin, np.inf), size=3, mode='nearest')
        zlow = np.where(np.isfinite(zlow), zlow, zmin)
        Z = laplace_fill(known, np.where(cov, zlow, 0.0), mask)
        Cc = laplace_fill(known, col, mask)
        V, Cv, Tt = grid_patch(bb0, cell, mask, Z - below, np.clip(Cc, 0, 255), self.O)
        return (V, Cv, Tt, int(hole.sum())) if len(Tt) else None

    # ---------------------------------------------------------------- binary payloads
    @staticmethod
    def pack(header, arrays):
        hb = json.dumps(header).encode()
        hb += b' ' * ((4 - (len(hb) + 8) % 4) % 4)
        out = [b'RSCK', np.uint32(len(hb)).tobytes(), hb]
        for a in arrays:
            b = np.ascontiguousarray(a).tobytes()
            out.append(b + b'\0' * ((4 - len(b) % 4) % 4))
        return b''.join(out)

    def chunk_bytes(self, key, lod):
        if lod:
            V, C, Tt = self.chunk_lod(key)
            return self.pack(dict(key=key, nv=len(V), nt=len(Tt), lod=1), [V, C, Tt])
        d = self.load(key)
        V, C = self.verts(key, preview=True)
        return self.pack(dict(key=key, nv=len(V), nt=len(d['T']), lod=0), [V, C, d['T'], self.Hcur(key)])

    def chunk_lod(self, key):
        import open3d as o3d
        ver = self.chunks[key].get('ver', 0)
        if key in self.lod and self.lod[key][0] == ver:
            return self.lod[key][1]
        d = self.load(key)
        keep = ~self.dmask(key)
        V, C = self.verts(key, preview=True)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)),
                                      o3d.utility.Vector3iVector(np.asarray(d['T'])[keep].astype(np.int32)))
        m.vertex_colors = o3d.utility.Vector3dVector(C / 255.0)
        m = m.simplify_vertex_clustering(0.4, o3d.geometry.SimplificationContraction.Average)
        res = (np.asarray(m.vertices, np.float32), (np.asarray(m.vertex_colors) * 255).round().astype(np.uint8),
               np.asarray(m.triangles, np.uint32))
        self.lod[key] = (ver, res)
        return res

    def mask_bytes(self, key, which='del'):
        if which == 'del':
            m = self.dmask(key)
        else:
            p = self.proposal
            m = np.zeros(self.chunks[key]['nt'], bool)
            if p and key in p['del']:
                m[p['del'][key]] = True
        return np.packbits(m, bitorder='little').tobytes()

    def patch_bytes(self, pid):
        p = self.patches.get(pid) or (self.proposal and {q['pid']: q for q in self.proposal['patches']}.get(pid))
        return self.pack(dict(pid=pid, nv=len(p['V']), nt=len(p['T'])), [p['V'], p['C'], p['T']])

    def patch_list(self):
        out = [dict(pid=pid, kind=p['kind'], label=p.get('label', ''), active=p['active'], key=p['key'], nt=int(len(p['T'])), preview=False)
               for pid, p in self.patches.items()]
        if self.proposal:
            out += [dict(pid=q['pid'], kind=q['kind'], label=q.get('label', ''), active=True, key=q['key'], nt=int(len(q['T'])), preview=True)
                    for q in self.proposal['patches']]
        return out

    # ---------------------------------------------------------------- edits, undo/redo
    def _touch(self, keys):
        self.version += 1
        for k in keys:
            self.chunks[k]['ver'] = self.chunks[k].get('ver', 0) + 1
        self.changes.append((self.version, sorted(set(keys))))
        self.changes = self.changes[-500:]

    def _apply(self, op, forward=True):
        z = np.load(os.path.join(EDITS, 'ops', op['id'] + '.npz'))
        keys = set()
        for name in z.files:
            kind, key = name.split(':', 1)
            if kind in ('del', 'undel'):
                self.dmask(key)[z[name]] = (kind == 'del') if forward else (kind != 'del')
            elif kind == 'vi':
                s = '1' if forward else '0'
                self._set_verts(key, z[name], z[f'vz{s}:{key}'], z[f'vc{s}:{key}'])
            else:
                continue
            keys.add(key)
        for pid in op.get('add', []):
            self.patches[pid]['active'] = forward
        for pid in op.get('remove', []):
            self.patches[pid]['active'] = not forward
        self._touch(keys)
        self._save(keys)
        return keys

    def commit(self, label, dele=None, undel=None, add=None, remove=None, verts=None, tag=None):
        """Record + apply an edit; only indices that actually change are stored (exact undo).
        verts = {key: (idx, new z rel O, new colour uint8)}. Edits with the same 'tag' as the last op
        (one brush stroke sent in pieces) are merged into it: one undo step."""
        arrays = {}
        for kind, sets in (('del', dele or {}), ('undel', undel or {})):
            for key, idx in sets.items():
                idx = np.unique(np.asarray(idx, np.int64))
                if not len(idx):
                    continue
                m = self.dmask(key)
                idx = idx[~m[idx]] if kind == 'del' else idx[m[idx]]
                if len(idx):
                    arrays[f'{kind}:{key}'] = idx.astype(np.int32)
        for key, (idx, z1, c1) in (verts or {}).items():
            self._varrays(key)
            idx = np.asarray(idx, np.int64)
            if not len(idx):
                continue
            arrays[f'vi:{key}'] = idx.astype(np.int32)
            arrays[f'vz0:{key}'] = self.vz[key][idx].copy(); arrays[f'vc0:{key}'] = self.vc[key][idx].copy()
            arrays[f'vz1:{key}'] = np.asarray(z1, np.float32); arrays[f'vc1:{key}'] = np.asarray(c1).astype(np.int16)
        if not arrays and not add and not remove:
            return dict(ok=True, changed=0, keys=[])
        last = self.ops[-1] if self.ops else None
        if tag and last and last.get('tag') == tag and not self.redo:
            # merge into the previous piece of the same stroke: keep its 'before' values, take the new 'after'
            zold = dict(np.load(os.path.join(EDITS, 'ops', last['id'] + '.npz')))
            for name in [n for n in arrays if n.startswith('vi:')]:
                key = name[3:]
                if name in zold:
                    i0, i1 = zold[name].astype(np.int64), arrays[name].astype(np.int64)
                    allidx = np.union1d(i0, i1)
                    def merged(field, first):
                        a = (zold if first else arrays)[f'{field}:{key}']
                        out = np.zeros((len(allidx),) + a.shape[1:], a.dtype)
                        # 'before': from the earlier piece where it has the vertex, else the current value
                        if field.endswith('0'):
                            cur = self.vz[key] if field.startswith('vz') else self.vc[key]
                            out[:] = cur[allidx]
                            out[np.searchsorted(allidx, i0)] = zold[f'{field}:{key}']
                        else:
                            out[:] = (self.vz[key] if field.startswith('vz') else self.vc[key])[allidx]
                            out[np.searchsorted(allidx, i1)] = arrays[f'{field}:{key}']
                        return out
                    for f in ('vz0', 'vc0', 'vz1', 'vc1'):
                        arrays[f'{f}:{key}'] = merged(f, f.endswith('0'))
                    arrays[name] = allidx.astype(np.int32)
            new_vkeys = {n[3:] for n in arrays if n.startswith('vi:')}
            for n, v in zold.items():
                kind, key = n.split(':', 1)
                if kind in ('del', 'undel'):
                    arrays[n] = np.union1d(v, arrays[n]).astype(np.int32) if n in arrays else v
                elif key not in new_vkeys:      # vertex edits of chunks this piece did not touch
                    arrays[n] = v
            op = last
            op['n'] = int(sum(len(v) for n, v in arrays.items() if n.split(':')[0] in ('del', 'undel', 'vi')))
            op['add'] = op.get('add', []) + (add or [])
            np.savez(os.path.join(EDITS, 'ops', op['id'] + '.npz'), **arrays)
            keys = set()
            for name in list(arrays):
                kind, key = name.split(':', 1)
                if kind == 'vi':
                    self._set_verts(key, arrays[name], arrays[f'vz1:{key}'], arrays[f'vc1:{key}'])
                elif kind == 'del':
                    self.dmask(key)[arrays[name]] = True
                elif kind == 'undel':
                    self.dmask(key)[arrays[name]] = False
                else:
                    continue
                keys.add(key)
            for pid in add or []:
                self.patches[pid]['active'] = True
            self._touch(keys); self._save(keys)
            return dict(ok=True, changed=op['n'], keys=sorted(keys))
        op = dict(id=f'{int(time.time() * 1000)}_{len(self.ops)}', label=label, add=add or [], remove=remove or [], tag=tag,
                  n=int(sum(len(v) for n, v in arrays.items() if n.split(':')[0] in ('del', 'undel', 'vi'))))
        np.savez(os.path.join(EDITS, 'ops', op['id'] + '.npz'), **arrays)
        self.ops.append(op); self.redo = []
        keys = self._apply(op, True)
        return dict(ok=True, changed=op['n'], keys=sorted(keys))

    def undo(self):
        if not self.ops:
            return dict(ok=False)
        op = self.ops.pop(); self.redo.append(op); self._apply(op, False)      # lists first: _apply saves them
        return dict(ok=True, label=op['label'])

    def redo_(self):
        if not self.redo:
            return dict(ok=False)
        op = self.redo.pop(); self.ops.append(op); self._apply(op, True)
        return dict(ok=True, label=op['label'])

    def _save(self, keys):
        for k in keys:
            np.save(os.path.join(EDITS, 'del', k + '.npy'), self.dmask(k))
            if k in self.vz:
                f = os.path.join(EDITS, 'vert', k + '.npz')
                idx = np.flatnonzero(np.isfinite(self.vz[k]) | (self.vc[k][:, 0] >= 0))
                if len(idx):
                    np.savez(f, idx=idx.astype(np.int32), z=self.vz[k][idx], c=self.vc[k][idx])
                elif os.path.exists(f):
                    os.remove(f)
        state = dict(patches={pid: dict(kind=p['kind'], key=p['key'], active=p['active'], label=p.get('label', ''))
                              for pid, p in self.patches.items()}, ops=self.ops, redo=self.redo)
        json.dump(state, open(os.path.join(EDITS, 'state.json.tmp'), 'w'))
        os.replace(os.path.join(EDITS, 'state.json.tmp'), os.path.join(EDITS, 'state.json'))

    def add_patch(self, kind, key, V, C, Tt, label):
        pid = f'{kind}_{int(time.time() * 1000)}_{len(self.patches)}'
        np.savez(os.path.join(EDITS, 'patches', pid + '.npz'), V=V, C=C, T=Tt)
        self.patches[pid] = dict(kind=kind, key=key, V=V, C=C, T=Tt, active=False, label=label)
        return pid

    def summary(self):
        ndel = int(sum(m.sum() for m in list(self.deleted.values())))
        nver = int(sum(np.isfinite(z).sum() for z in list(self.vz.values())))
        act = [p for p in list(self.patches.values()) if p['active']]
        return dict(deleted=ndel, moved=nver, patches=len(act), patch_tris=int(sum(len(p['T']) for p in act)),
                    total=int(sum(c['nt'] for c in self.chunks.values())),
                    undo=[o['label'] for o in self.ops[-15:]], redo=[o['label'] for o in self.redo[-15:]])

    # ---------------------------------------------------------------- corridor (lane edges)
    def corridor(self):
        """Lateral position of the left/right road edges along the path (+ = right): annotated lane edges where
        drawn, else the project's default half-width either side of the path."""
        if self.corr is not None and self.corr[0] == self._lanes_stamp():
            return self.corr[1]
        s = np.arange(0, self.route.arc[-1] + 0.5, 0.5)
        hw = float(PROJECT.settings.get('corridor_half', 4.0))
        lo_s, hi_s = np.full(len(s), -hw), np.full(len(s), hw)
        ann_j = os.path.join(ANN, 'annotation.json')
        if os.path.exists(ann_j):
            try:
                aj = json.load(open(ann_j))
                for typ, dst in (('left_edge', lo_s), ('right_edge', hi_s)):
                    pts = [resample(catmull_rom(ln['nodes'], ln.get('smooth', True)), 0.25)
                           for ln in aj.get('lines', []) if ln.get('type') == typ and len(ln.get('nodes', [])) >= 2 and ln.get('visible', True)]
                    if pts:
                        X = np.concatenate(pts)
                        a, l = self.route.project(X)
                        o = np.argsort(a); a, l = a[o], l[o]
                        a, u = np.unique(a, return_index=True); l = l[u]
                        i = np.clip(np.searchsorted(a, s), 0, len(a) - 1)
                        near = np.minimum(np.abs(a[i] - s), np.abs(a[np.clip(i - 1, 0, len(a) - 1)] - s)) < 1.5
                        dst[near] = np.interp(s[near], a, l)
            except Exception:
                pass
        res = (s, np.minimum(lo_s, hi_s), np.maximum(lo_s, hi_s))
        self.corr = (self._lanes_stamp(), res)
        return res

    def _lanes_stamp(self):
        ann_j = os.path.join(ANN, 'annotation.json')
        return os.path.getmtime(ann_j) if os.path.exists(ann_j) else 0

    def edges_at(self, arc):
        s, lo, hi = self.corridor()
        return np.interp(arc, s, lo), np.interp(arc, s, hi)

    # ---------------------------------------------------------------- automatic tools (proposals)
    def _progress(self, msg, frac):
        self.status.update(msg=msg, progress=frac)

    def propose(self, tool, prm):
        arc0, arc1 = float(prm.get('arc0', 0)), float(prm.get('arc1', 1e9))
        keys = self.keys_in(arc0, arc1)
        dele, patches, info = {}, [], {}
        if tool == 'ghost':
            m, hmin, hmax = float(prm.get('margin', 0.5)), float(prm.get('hmin', 0.3)), float(prm.get('hmax', 4.5))
            for n, k in enumerate(keys):
                self._progress(f'ghost obstacles: {k}', n / len(keys))
                d = self.load(k); arc, lat, H = np.asarray(d['arc']), np.asarray(d['lat']), self.Hcur(k) / 100
                lo, hi = self.edges_at(arc)
                sel = (arc >= arc0) & (arc <= arc1) & (lat > lo - m) & (lat < hi + m) & (H > hmin) & (H < hmax) & ~self.dmask(k)
                if sel.any():
                    dele[k] = np.flatnonzero(sel)
            label = f'ghost obstacles over the road ({hmin}-{hmax} m, edges +{m} m)'
        elif tool == 'floaters':
            # pieces hanging in the air: connected groups of triangles above 'hmin' that never come down
            # closer than 'gap' to the ground and stay below 'top' (so trees / bushes standing on the ground stay)
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            hmin, gap, top, maxlat = float(prm.get('hmin', 0.3)), float(prm.get('gap', 0.5)), float(prm.get('top', 6.0)), float(prm.get('maxlat', 15))
            for n, k in enumerate(keys):
                self._progress(f'floaters: {k}', n / len(keys))
                d = self.load(k); arc, lat, H = np.asarray(d['arc']), np.asarray(d['lat']), self.Hcur(k) / 100
                cand = np.flatnonzero((H > hmin) & ~self.dmask(k))
                if not len(cand):
                    continue
                Tt = np.asarray(d['T'])[cand].astype(np.int64)
                nv = int(Tt.max()) + 1
                A = coo_matrix((np.ones(3 * len(cand)), (np.repeat(np.arange(len(cand)), 3), Tt.ravel())), shape=(len(cand), nv)).tocsr()
                nc, lab = connected_components((A @ A.T).tocsr(), directed=False)
                lo_h = np.full(nc, np.inf); np.minimum.at(lo_h, lab, H[cand])
                hi_h = np.full(nc, -np.inf); np.maximum.at(hi_h, lab, H[cand])
                floating = (lo_h > gap) & (hi_h < top)
                sel = cand[floating[lab] & (arc[cand] >= arc0) & (arc[cand] <= arc1) & (np.abs(lat[cand]) < maxlat)]
                if len(sel):
                    dele[k] = sel
            label = f'floaters (pieces hanging > {gap} m above the ground, below {top} m)'
        elif tool == 'islands':
            mint, maxlat = int(prm.get('min_tris', 2000)), float(prm.get('maxlat', 40))
            for n, k in enumerate(keys):
                self._progress(f'islands: {k}', n / len(keys))
                d = self.load(k); arc, lat = np.asarray(d['arc']), np.asarray(d['lat'])
                sel = (arc >= arc0) & (arc <= arc1) & (np.abs(lat) < maxlat) & (np.asarray(d['csize']) < mint) & ~self.dmask(k)
                if sel.any():
                    dele[k] = np.flatnonzero(sel)
            label = f'disconnected pieces (< {mint} triangles)'
        elif tool == 'road':
            dele, patches = self.road_level(keys, arc0, arc1, prm)
            label = 'levelled road surface'
        elif tool == 'skirt':
            patches = self.skirt(keys, arc0, arc1, prm)
            label = 'off-road safety ground'
        elif tool == 'road_seg':
            m = float(prm.get('margin', 0.0))
            hmin, hmax = float(prm.get('hmin', -0.8)), float(prm.get('hmax', 0.6))
            for n, k in enumerate(keys):
                self._progress(f'segment road: {k}', n / len(keys))
                d = self.load(k); arc, lat, H = np.asarray(d['arc']), np.asarray(d['lat']), self.Hcur(k) / 100.0
                lo, hi = self.edges_at(arc)
                sel = (arc >= arc0) & (arc <= arc1) & (lat >= lo - m) & (lat <= hi + m) & (H >= hmin) & (H <= hmax) & ~self.dmask(k)
                if sel.any():
                    dele[k] = np.flatnonzero(sel)
            label = f'Class Road segmentation preview (margin {m:+.2f} m, height <= {hmax:.2f} m)'
        else:
            raise ValueError(tool)
        n = int(sum(len(v) for v in dele.values()))
        self.proposal = {'tool': tool, 'label': label, 'del': dele, 'patches': patches, 'prm': prm,
                         'stats': dict(delete=n, patch_tris=int(sum(len(p['T']) for p in patches)), chunks=len(set(dele) | {p['key'] for p in patches}))}
        self.version += 1
        self.changes.append((self.version, sorted(set(dele) | {p['key'] for p in patches})))
        return {**self.proposal['stats'], 'label': label}

    def road_model(self, arc0, arc1):
        """Levelled road surface: per metre a + b*lat (crossfall) from the ground-surface gaussians, smoothed."""
        from scipy.ndimage import gaussian_filter1d
        if not hasattr(self, '_surf'):
            S = np.load(os.path.join(TILES, 'surface_xyz.npy'))
            a, l = self.route.project(S[:, :2])
            self._surf = (S, a, l)
        S, a, l = self._surf
        s = np.arange(max(0, arc0 - 10), min(self.route.arc[-1], arc1 + 10) + 1, 1.0)
        lo, hi = self.edges_at(s)
        k = (a >= s[0] - 1) & (a <= s[-1] + 1)
        Sa, Sl, Sz = a[k], l[k], S[k, 2]
        loi, hii = self.edges_at(Sa)
        k = (Sl > loi + 0.1) & (Sl < hii - 0.1)
        Sa, Sl, Sz = Sa[k], Sl[k], Sz[k]
        o = np.argsort(Sa); Sa, Sl, Sz = Sa[o], Sl[o], Sz[o]
        A = np.full(len(s), np.nan); B = np.zeros(len(s))
        for i, si in enumerate(s):
            b0, b1 = np.searchsorted(Sa, si - 0.75), np.searchsorted(Sa, si + 0.75)
            if b1 - b0 < 20:
                continue
            x, z = Sl[b0:b1], Sz[b0:b1]
            w = np.ones(len(x))
            for _ in range(3):  # robust line fit across the road
                X = np.c_[np.ones(len(x)), x - (lo[i] + hi[i]) / 2]
                coef = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)[0]
                r = np.abs(z - X @ coef); w = 1 / np.maximum(r / 0.05, 1)
            A[i], B[i] = coef
        ok = np.isfinite(A)
        if ok.sum() < 2:
            return None
        A = np.interp(s, s[ok], A[ok]); B = np.where(ok, B, 0)
        A, B = gaussian_filter1d(A, 2.0, mode='nearest'), gaussian_filter1d(B, 4.0, mode='nearest')
        mid = (lo + hi) / 2
        return lambda arc, lat: np.interp(arc, s, A) + np.interp(arc, s, B) * (lat - np.interp(arc, s, mid))

    def road_level(self, keys, arc0, arc1, prm):
        m, cell, blend = float(prm.get('margin', 0.3)), float(prm.get('cell', 0.15)), float(prm.get('blend', 0.5))
        band_lo, band_hi = float(prm.get('band_lo', -0.6)), float(prm.get('band_hi', 0.3))
        model = self.road_model(arc0, arc1)
        dele, patches = {}, []
        for n, k in enumerate(keys):
            self._progress(f'road surface: {k}', n / len(keys))
            c = self.chunks[k]
            a0, a1 = max(arc0, c['arc0']), min(arc1, c['arc1'])
            if a1 <= a0:
                continue
            d = self.load(k); arc, lat, H = np.asarray(d['arc']), np.asarray(d['lat']), self.Hcur(k) / 100
            lo, hi = self.edges_at(arc)
            cz = self.centroids(k)[:, 2]
            rz = model(arc, lat)
            sel = (arc >= a0) & (arc <= a1) & (lat > lo - m) & (lat < hi + m) & (cz - rz > band_lo) & (cz - rz < band_hi) & ~self.dmask(k)
            if sel.any():
                dele[k] = np.flatnonzero(sel)
            # grid over this chunk's stretch of corridor; the patch's cells are owned by this chunk's arc range
            ss = np.arange(a0, a1 + 1)
            elo, ehi = self.edges_at(ss)
            P = self.route.at(ss)[:, :2]
            bb0, bb1 = P.min(0) - 8, P.max(0) + 8
            nx, ny = int((bb1[0] - bb0[0]) / cell) + 1, int((bb1[1] - bb0[1]) / cell) + 1
            gx, gy = np.meshgrid(bb0[0] + (np.arange(nx) + 0.5) * cell, bb0[1] + (np.arange(ny) + 0.5) * cell, indexing='ij')
            g = np.c_[gx.ravel(), gy.ravel()]
            ga, gl = self.route.project(g)
            glo, ghi = self.edges_at(ga)
            inside = (ga >= a0 - 4 * cell) & (ga <= a1 + 4 * cell) & (gl > glo - m - max(3 * cell, 0.15)) & (gl < ghi + m + max(3 * cell, 0.15))
            if inside.sum() < 4:
                continue
            Z = np.full(len(g), np.nan)
            Z[inside] = model(ga[inside], gl[inside])
            # blend toward the neighbours (ground surface) over the outer 'blend' metres of the strip
            edge_d = np.minimum(gl - (glo - m), (ghi + m) - gl)
            w = np.clip(edge_d / max(blend, 1e-3), 0, 1)
            Zg = self.ground_at(g)
            Z = np.where(inside, w * Z + (1 - w) * Zg, np.nan)
            mask = inside.reshape(nx, ny)
            Cc = self.ortho.area(g, cell).reshape(nx, ny, 3)
            V, Cv, Tt = grid_patch(bb0, cell, mask, Z.reshape(nx, ny), Cc, self.O)
            patches.append(dict(pid=f'prev_road_{k}', kind='road', key=k, V=V, C=Cv, T=Tt, label=f'road {k}'))
        return dele, patches

    def skirt(self, keys, arc0, arc1, prm):
        width, cell, below = float(prm.get('width', 30)), float(prm.get('cell', 0.5)), float(prm.get('below', 0.05))
        only_missing = bool(prm.get('only_missing', True))
        patches = []
        for n, k in enumerate(keys):
            self._progress(f'off-road ground: {k}', n / len(keys))
            c = self.chunks[k]
            a0, a1 = max(arc0, c['arc0']), min(arc1, c['arc1'])
            if a1 <= a0 or c['part'] != self._owner_part(k):
                continue
            P = self.route.at(np.arange(a0, a1 + 1))[:, :2]
            bb0, bb1 = P.min(0) - width - 2, P.max(0) + width + 2
            nx, ny = int((bb1[0] - bb0[0]) / cell) + 1, int((bb1[1] - bb0[1]) / cell) + 1
            gx, gy = np.meshgrid(bb0[0] + (np.arange(nx) + 0.5) * cell, bb0[1] + (np.arange(ny) + 0.5) * cell, indexing='ij')
            g = np.c_[gx.ravel(), gy.ravel()]
            ga, gl = self.route.project(g)
            inside = (ga >= a0 - cell) & (ga <= a1 + cell) & (np.abs(gl) < width)
            mask = inside.reshape(nx, ny)
            if only_missing:  # only where no (kept) ground-like mesh triangle is near
                cov = np.zeros((nx, ny), bool)
                for kk in self.keys_near(bb0, bb1):
                    d = self.load(kk)
                    keep = ~self.dmask(kk) & (np.abs(self.Hcur(kk)) < 150)
                    ce = self.centroids(kk)[keep]
                    ii = ((ce[:, 0] - bb0[0]) / cell).astype(int); jj = ((ce[:, 1] - bb0[1]) / cell).astype(int)
                    ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                    cov[ii[ok], jj[ok]] = True
                for p in self.patches.values():
                    if p['active'] and p['kind'] in ('road', 'fill', 'hole'):
                        ce = p['V'][p['T']].mean(1) + self.O
                        ii = ((ce[:, 0] - bb0[0]) / cell).astype(int); jj = ((ce[:, 1] - bb0[1]) / cell).astype(int)
                        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                        cov[ii[ok], jj[ok]] = True
                mask = mask & self._dilate(~cov, 1)  # no mesh ground here, plus one cell of overlap under the mesh
            if mask.sum() < 4:
                continue
            Z = (self.ground_at(g) - below).reshape(nx, ny)
            Cc = self.ortho.colours(g).reshape(nx, ny, 3)
            V, Cv, Tt = grid_patch(bb0, cell, mask, Z, Cc, self.O)
            if len(Tt):
                patches.append(dict(pid=f'prev_skirt_{k}', kind='skirt', key=k, V=V, C=Cv, T=Tt, label=f'ground {k}'))
        return patches

    @staticmethod
    def _dilate(m, it):
        from scipy.ndimage import binary_dilation
        return binary_dilation(m, iterations=it)

    def _owner_part(self, key):
        """Skirt patches are generated once per route bin: by the first part that has a chunk in that bin."""
        b = self.chunks[key]['bin']
        return min(c['part'] for c in self.chunks.values() if c['bin'] == b)

    # ---------------------------------------------------------------- manual tools
    def blob(self, key, tri, hprotect):
        """Triangles connected to 'tri' (shared vertices) that are more than hprotect above the ground."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order
        d = self.load(key)
        Tt = np.asarray(d['T']).astype(np.int64); H = self.Hcur(k) / 100
        ok = (H > hprotect) & ~self.dmask(key)
        if not ok[tri]:
            comp = np.asarray(d['comp']); sel = np.flatnonzero((comp == comp[tri]) & ~self.dmask(key))
            if len(sel) > 200000:
                return None, 'That triangle is on the ground: nothing to pick (raise the protect height or use the brush)'
            return {key: sel}, f'whole connected piece: {len(sel):,} triangles'
        cand = np.flatnonzero(ok)
        pos = -np.ones(len(Tt), np.int64); pos[cand] = np.arange(len(cand))
        nv = int(Tt.max()) + 1
        # triangle graph via shared vertices: tri -> vertex -> tri
        A = coo_matrix((np.ones(3 * len(cand)), (np.repeat(np.arange(len(cand)), 3), Tt[cand].ravel())), shape=(len(cand), nv)).tocsr()
        G = (A @ A.T).tocsr()
        order = breadth_first_order(G, pos[tri], directed=False, return_predecessors=False)
        return {key: cand[order]}, f'{len(order):,} connected triangles above {hprotect} m'

    @staticmethod
    def _bilinear(G, bb0, cell, xy):
        """Sample a cell-centred grid at points (clamped at the border)."""
        nx, ny = G.shape[:2]
        fx = np.clip((xy[:, 0] - bb0[0]) / cell - 0.5, 0, nx - 1.001); fy = np.clip((xy[:, 1] - bb0[1]) / cell - 0.5, 0, ny - 1.001)
        i0, j0 = fx.astype(int), fy.astype(int); ax, ay = fx - i0, fy - j0
        i1, j1 = np.minimum(i0 + 1, nx - 1), np.minimum(j0 + 1, ny - 1)
        if G.ndim == 3:
            ax, ay = ax[:, None], ay[:, None]
        return G[i0, j0] * (1 - ax) * (1 - ay) + G[i1, j0] * ax * (1 - ay) + G[i0, j1] * (1 - ax) * ay + G[i1, j1] * ax * ay

    def _cell_stats(self, pts, cols, bb0, cell, nx, ny, q, sel):
        """Per grid cell: q-quantile height of the selected vertices and the mean colour of those near it."""
        ii = ((pts[:, 0] - bb0[0]) / cell).astype(int); jj = ((pts[:, 1] - bb0[1]) / cell).astype(int)
        ok = sel & (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        known = np.zeros((nx, ny), bool); Z = np.zeros((nx, ny)); C = np.zeros((nx, ny, 3))
        lin = ii[ok] * ny + jj[ok]
        if not len(lin):
            return known, Z, C
        z, c = pts[ok, 2], cols[ok]
        order = np.lexsort((z, lin)); ls = lin[order]
        st = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]]); cnt = np.diff(np.r_[st, len(ls)])
        zq = z[order][st + (q * (cnt - 1)).astype(int)]
        kk = np.unravel_index(ls[st], (nx, ny))
        known[kk] = True; Z[kk] = zq
        near = np.abs(z - Z.ravel()[lin]) < 0.12           # colour of the surface, not of spikes / leaves
        csum = np.zeros((nx * ny, 3)); n = np.zeros(nx * ny)
        np.add.at(csum, lin[near], c[near]); np.add.at(n, lin[near], 1)
        C = (csum / np.maximum(n, 1)[:, None]).reshape(nx, ny, 3)
        return known, Z, C

    def fill(self, poly, prm):
        """Contour fill. The existing surface inside the contour is reshaped in place (levelled plane or smooth
        membrane fitted to the ground just outside), feathered to zero at the contour, so there is no seam.
        Real holes are closed by a surface just under the mesh that overlaps it. Nothing outside changes."""
        from scipy.ndimage import distance_transform_edt
        import cv2
        poly = np.asarray(poly, float)
        mode, colour, remove = prm.get('mode', 'level'), prm.get('colour', 'keep'), prm.get('remove', 'ground')
        clear_h, flat_h, blend = float(prm.get('clear_h', 4.5)), float(prm.get('flat_h', 0.6)), float(prm.get('blend', 0.6))
        bb0, bb1 = poly.min(0) - 1.5, poly.max(0) + 1.5
        area = np.prod(bb1 - bb0)
        cell = float(prm.get('cell', 0)) or float(np.clip(math.sqrt(area / 150000), 0.05, 0.5))
        nx, ny = int((bb1[0] - bb0[0]) / cell) + 1, int((bb1[1] - bb0[1]) / cell) + 1
        raster = np.zeros((ny, nx), np.uint8)
        cv2.fillPoly(raster, [np.round((poly - bb0) / cell - 0.5).astype(np.int32)], 1)
        inside = raster.T.astype(bool)
        ring = self._dilate(inside, max(2, int(round(0.6 / cell)))) & ~inside
        gx, gy = np.meshgrid(bb0[0] + (np.arange(nx) + 0.5) * cell, bb0[1] + (np.arange(ny) + 0.5) * cell, indexing='ij')
        Zg = self.ground_at(np.c_[gx.ravel(), gy.ravel()]).reshape(nx, ny)
        G = self.gather(bb0, bb1)
        pts = np.concatenate([g[2] for g in G]) if G else np.zeros((0, 3))
        cols = np.concatenate([g[3] for g in G]) if G else np.zeros((0, 3))
        ii = np.clip(((pts[:, 0] - bb0[0]) / cell).astype(int), 0, nx - 1); jj = np.clip(((pts[:, 1] - bb0[1]) / cell).astype(int), 0, ny - 1)
        ground_like = np.abs(pts[:, 2] - Zg[ii, jj]) < 1.0
        known, Zk, Ck = self._cell_stats(pts, cols, bb0, cell, nx, ny, 0.3, ground_like & ring[ii, jj])
        if known.sum() < 3:      # no mesh around: the ground grid on the ring
            known = ring.copy(); Zk = Zg.copy()
        everywhere = np.ones((nx, ny), bool)
        Zs = laplace_fill(known, Zk, everywhere)
        dist_in = distance_transform_edt(inside) * cell
        if mode == 'level':
            kz = known; zv = Zk[kz]
            X = np.c_[np.ones(len(zv)), gx[kz], gy[kz]]
            w = np.ones(len(zv))
            for _ in range(4):
                coef = np.linalg.lstsq(X * w[:, None], zv * w, rcond=None)[0]
                r = np.abs(zv - X @ coef); w = 1 / np.maximum(r / 0.05, 1)
            plane = coef[0] + coef[1] * gx + coef[2] * gy
            wgt = np.clip(dist_in / max(blend, cell), 0, 1)
            wgt = wgt * wgt * (3 - 2 * wgt)
            Zs = wgt * plane + (1 - wgt) * Zs
        ck = known & (Ck.sum(-1) > 0)
        Cb = laplace_fill(ck, Ck, everywhere) if ck.sum() >= 3 else None
        if colour in ('ortho', 'keep'):
            Cb = self.ortho.area(np.c_[gx.ravel(), gy.ravel()], cell).reshape(nx, ny, 3).astype(float)
        # feather: full effect from 'feather' inside the contour, none at the contour itself
        feather = float(np.clip(0.5 * blend, 2 * cell, 0.35))
        F = np.clip(dist_in / feather, 0, 1); F = F * F * (3 - 2 * F)
        verts, dele = {}, {}
        moved_n = 0
        for k, idx, P, C in G:
            w = self._bilinear(F, bb0, cell, P[:, :2])
            ins = inside[np.clip(((P[:, 0] - bb0[0]) / cell).astype(int), 0, nx - 1), np.clip(((P[:, 1] - bb0[1]) / cell).astype(int), 0, ny - 1)]
            w = np.where(ins, w, 0.0)
            zt = self._bilinear(Zs, bb0, cell, P[:, :2])
            dz = P[:, 2] - zt
            mv = (w > 0) & (np.abs(dz) < flat_h) if remove != 'none' else np.zeros(len(P), bool)
            if remove == 'all':      # taller things inside (bushes, ghost blobs, poles) are removed, not squashed
                d = self.load(k); Tt = np.asarray(d['T'])
                tall = np.zeros(self.chunks[k]['nv'], bool); tall[idx[ins & (dz >= flat_h) & (dz < clear_h)]] = True
                sel = np.flatnonzero(tall[Tt].any(1) & ~self.dmask(k))
                if len(sel):
                    dele[k] = sel
            if not mv.any():
                continue
            wm = w[mv]
            z1 = P[mv, 2] + wm * (zt[mv] - P[mv, 2])
            if Cb is not None and colour in ('blend', 'ortho'):
                a = wm
            elif Cb is not None:     # keep the texture; spikes and odd-coloured blobs take the blended colour
                Cn = self._bilinear(Cb, bb0, cell, P[mv, :2])
                a = wm * np.clip((np.abs(dz[mv]) - 0.08) / 0.25, 0, 1)
            else:
                a = np.zeros(len(wm))
            c1 = C[mv] + a[:, None] * (self._bilinear(Cb, bb0, cell, P[mv, :2]) - C[mv]) if Cb is not None else C[mv]
            order = np.argsort(idx[mv])
            verts[k] = (idx[mv][order], (z1 - self.O[2]).astype(np.float32)[order], np.clip(np.round(c1), 0, 255).astype(np.uint8)[order])
            moved_n += int(mv.sum())
        cxy = poly.mean(0)
        arc, _ = self.route.project(cxy[None])
        key = min(self.chunks, key=lambda k: abs((self.chunks[k]['arc0'] + self.chunks[k]['arc1']) / 2 - arc[0]))
        patches = []
        up = self.under_patch(bb0, cell, nx, ny, inside, override=verts, skip=dele, full=True, zref=Zs)
        if up:
            V, Cv, Tt, nh = up
            patches.append(dict(pid=f'prev_fill_{int(time.time() * 1000)}', kind='hole', key=key, V=V, C=Cv, T=Tt,
                                label=f'under-surface of fill ({nh} open cells)'))
        label = f'fill {mode} {len(poly)}-point contour'
        n = int(sum(len(v) for v in dele.values()))
        self.proposal = {'tool': 'fill', 'label': label, 'del': dele, 'patches': patches, 'verts': verts,
                         'stats': dict(delete=n, moved=moved_n, patch_tris=int(sum(len(q['T']) for q in patches)), cell=round(cell, 3),
                                       chunks=len(set(dele) | set(verts) | {key}))}
        for k in verts:
            self._varrays(k); self.touched[k][verts[k][0]] = True
        self.version += 1
        self.changes.append((self.version, sorted(set(dele) | set(verts) | {key})))
        return {**self.proposal['stats'], 'label': label}

    def smooth(self, pts, prm, stroke=None, final=False, close_only=False):
        """Smoothing brush: vertices within 'radius' of the stroke are pulled toward a robust smooth surface
        (median of the neighbourhood, then Gaussian) with a soft falloff - spikes and pits flatten out, the slope
        of the ground stays. Spikes take the colour of the surrounding surface. On the last piece of a stroke,
        holes under the brush are closed."""
        from scipy.ndimage import median_filter, gaussian_filter
        P = np.asarray(pts, float).reshape(-1, 3)
        if not len(P):
            return dict(ok=True, changed=0, keys=[])
        r, strength, hr = float(prm.get('radius', 0.6)), float(prm.get('strength', 0.5)), float(prm.get('range', 0.8))
        colours, close = prm.get('colour', 'spikes'), bool(prm.get('close_holes', True))
        scale = float(prm.get('scale', 0)) or r
        cell = float(np.clip(r / 6, 0.03, 0.2))
        pad = r + 2 * scale + 0.5
        bb0, bb1 = P[:, :2].min(0) - pad, P[:, :2].max(0) + pad
        nx, ny = int((bb1[0] - bb0[0]) / cell) + 1, int((bb1[1] - bb0[1]) / cell) + 1
        G = self.gather(bb0, bb1)
        if not G:
            return dict(ok=True, changed=0, keys=[])
        allp = np.concatenate([g[2] for g in G]); allc = np.concatenate([g[3] for g in G])
        # reference height: the stroke's surface where the brush is (not the canopy above / a pit below)
        zref = self.ground_at(allp[:, :2])
        sp = P[:, :2]
        dd = ((allp[:, None, :2] - sp[None]) ** 2).sum(-1) if len(sp) * len(allp) < 4e7 else None
        if dd is not None:
            zref = P[np.argmin(dd, 1), 2]
        sel = np.abs(allp[:, 2] - zref) < hr + 0.5
        known, Zk, Ck = self._cell_stats(allp, allc, bb0, cell, nx, ny, 0.5, sel)
        if known.sum() < 3:
            return dict(ok=True, changed=0, keys=[])
        everywhere = np.ones((nx, ny), bool)
        Z = laplace_fill(known, Zk, everywhere)
        m = max(3, int(round(0.5 * scale / cell)) | 1)
        Z = median_filter(Z, size=m, mode='nearest')
        S = gaussian_filter(Z, scale / 3 / cell, mode='nearest')
        ck = known & (Ck.sum(-1) > 0)
        Cs = None
        if ck.sum() >= 3:   # neighbourhood colour, robust to small odd-coloured blobs (median over ~1.5 radius)
            mc = max(5, int(round(1.5 * scale / cell)) | 1)
            Cs = median_filter(laplace_fill(ck, Ck, everywhere), size=(mc, mc, 1), mode='nearest')
            Cs = gaussian_filter(Cs, (scale / 3 / cell, scale / 3 / cell, 0), mode='nearest')
        # distance of each vertex to the stroke polyline
        def seg_dist(xy):
            if len(P) == 1:
                return np.hypot(*(xy - P[0, :2]).T)
            a, b = P[:-1, :2], P[1:, :2]
            out = np.full(len(xy), np.inf)
            for s in range(len(a)):
                v = b[s] - a[s]; L2 = max(v @ v, 1e-12)
                t = np.clip(((xy - a[s]) @ v) / L2, 0, 1)
                out = np.minimum(out, np.hypot(*(xy - a[s] - t[:, None] * v).T))
            return out
        verts = {}
        foot = np.zeros((nx, ny), bool)
        n = 0
        for k, idx, Pk, Ck_ in ([] if close_only else G):
            d = seg_dist(Pk[:, :2])
            near = d < r
            if not near.any():
                continue
            zt = self._bilinear(S, bb0, cell, Pk[near, :2])
            dz = Pk[near, 2] - zt
            u = d[near] / r
            w = strength * (1 - u * u) ** 2
            mv = np.abs(dz) < hr
            if not mv.any():
                continue
            w, dz, zt = w[mv], dz[mv], zt[mv]
            vi = idx[near][mv]; Pm = Pk[near][mv]; Cm = Ck_[near][mv]
            z1 = Pm[:, 2] + w * (zt - Pm[:, 2])
            if colours == 'ortho':
                c1 = self.ortho.area(Pm[:, :2], 0.05).astype(float); c1 = Cm + np.minimum(1, 2 * w)[:, None] * (c1 - Cm)
            elif colours == 'spikes':   # only geometric spikes change colour, to the splat's colour there (paint is kept)
                a = np.clip((np.abs(dz) - 0.06) / 0.2, 0, 1) * np.minimum(1, 2 * w)
                c1 = Cm + a[:, None] * (self.ortho.area(Pm[:, :2], 0.05).astype(float) - Cm)
            elif Cs is not None:
                Cn = self._bilinear(Cs, bb0, cell, Pm[:, :2])
                if colours == 'smooth':
                    a = w
                elif colours == 'spikes':   # spikes and blobs of a clearly different colour take the surrounding colour
                    odd = np.maximum(np.clip((np.abs(dz) - 0.06) / 0.2, 0, 1), np.clip((np.abs(Cm - Cn).mean(1) - 30) / 40, 0, 1))
                    a = odd * np.minimum(1, 2 * w)
                else:
                    a = np.zeros(len(w))
                c1 = Cm + a[:, None] * (Cn - Cm)
            else:
                c1 = Cm
            o = np.argsort(vi)
            verts[k] = (vi[o], (z1 - self.O[2]).astype(np.float32)[o], np.clip(np.round(c1), 0, 255).astype(np.uint8)[o])
            n += len(vi)
        add = []
        if final and close:
            gx, gy = np.meshgrid(bb0[0] + (np.arange(nx) + 0.5) * cell, bb0[1] + (np.arange(ny) + 0.5) * cell, indexing='ij')
            foot = (seg_dist(np.c_[gx.ravel(), gy.ravel()]) < r).reshape(nx, ny)
            # any opening under the stroke (checked on a 2x finer grid, so slivers count) -> surface under the footprint
            fx, fy = np.meshgrid(bb0[0] + (np.arange(2 * nx) + 0.5) * cell / 2, bb0[1] + (np.arange(2 * ny) + 0.5) * cell / 2, indexing='ij')
            ffoot = (seg_dist(np.c_[fx.ravel(), fy.ravel()]) < r).reshape(2 * nx, 2 * ny)
            fcov = self.coverage(bb0, cell / 2, 2 * nx, 2 * ny, override=verts, zref=np.repeat(np.repeat(S, 2, 0), 2, 1), band=hr)[0]
            up = self.under_patch(bb0, cell, nx, ny, foot, override=verts, full=True, zref=S) if (ffoot & ~fcov).any() else None
            if up:
                V, Cv, Tt, nh = up
                arc, _ = self.route.project(P[:1, :2])
                key = min(self.chunks, key=lambda k: abs((self.chunks[k]['arc0'] + self.chunks[k]['arc1']) / 2 - arc[0]))
                add = [self.add_patch('hole', key, V, Cv, Tt, f'hole under smoothing ({nh} cells)')]
        if not verts and not add:
            return dict(ok=True, changed=0, keys=[])
        res = self.commit(f'smooth brush r {r:g} m', verts=verts, add=add, tag=stroke)
        if add:
            self._touch([self.patches[add[0]]['key']])
        return dict(res, holes=len(add))

    def crop(self, poly, mode='outside', zmin=None, zmax=None):
        """Crop by a polygon (z-up xy): delete triangles 'outside' (keep the inside) or 'inside' it; optional height band
        (zmin / zmax above the ground) limits what is deleted. Becomes a proposal (accept / discard)."""
        from matplotlib.path import Path as MPath
        poly = np.asarray(poly, float); pp = MPath(poly)
        bb0, bb1 = poly.min(0), poly.max(0)
        dele = {}
        for k in self.chunks:
            ce = self.centroids(k)
            ins = np.zeros(len(ce), bool)
            box = (ce[:, 0] >= bb0[0]) & (ce[:, 0] <= bb1[0]) & (ce[:, 1] >= bb0[1]) & (ce[:, 1] <= bb1[1])
            if box.any():
                ins[box] = pp.contains_points(ce[box, :2])
            sel = (~ins if mode == 'outside' else ins) & ~self.dmask(k)
            if zmin is not None or zmax is not None:
                H = self.Hcur(k) / 100
                sel &= (H >= (zmin if zmin is not None else -1e9)) & (H <= (zmax if zmax is not None else 1e9))
            if sel.any():
                dele[k] = np.flatnonzero(sel)
        n = int(sum(len(v) for v in dele.values()))
        self.proposal = {'tool': 'crop', 'label': f'crop ({"keep inside" if mode == "outside" else "remove inside"})', 'del': dele, 'patches': [],
                         'stats': dict(delete=n, patch_tris=0, chunks=len(dele))}
        self._touch(list(dele))
        return dict(ok=True, **self.proposal['stats'])

    def recolour_ground(self, hmax=0.3, poly=None):
        """Re-project colour from the orthophoto onto ground-level vertices (all, or inside a polygon); heights kept. One undoable op."""
        if poly is not None:
            from matplotlib.path import Path as MPath
            pp = MPath(np.asarray(poly, float)); hmax = 2.0
        from scipy.spatial import cKDTree
        S = np.load(os.path.join(TILES, 'surface_xyz.npy'))
        kd = cKDTree(S[:, :2]); verts, n = {}, 0
        for i, k in enumerate(sorted(self.chunks)):
            self._progress(f'recolour {k}', i / len(self.chunks))
            d = self.load(k); Tt = np.asarray(d['T']); keep = ~self.dmask(k); H = self.Hcur(k) / 100
            g = np.unique(Tt[keep & (np.abs(H) < hmax)])
            if not len(g):
                continue
            V, _ = self.verts(k); P = V[g] + self.O
            ok = np.isfinite(kd.query(P[:, :2], distance_upper_bound=0.10)[0])
            if poly is not None:
                ok &= pp.contains_points(P[:, :2])
            g = g[ok]
            if len(g):
                verts[k] = (g, V[g, 2].astype(np.float32), self.ortho.area(P[ok, :2], 0.05)); n += len(g)
        lab = 're-project colour from the orthophoto' + (' (selection)' if poly is not None else ' (all ground)')
        return self.commit(lab, verts=verts) if verts else dict(ok=True, changed=0)

    def accept(self):
        p = self.proposal
        if not p:
            return dict(ok=False)
        if p.get('tool') == 'road_seg':
            keys = set(p['del'])
            self.proposal = None
            self._touch(keys)
            return dict(ok=True, changed=0, keys=sorted(keys), road_seg=True)
        add = [self.add_patch(q['kind'], q['key'], q['V'], q['C'], q['T'], q.get('label', '')) for q in p['patches']]
        keys = set(p['del']) | {q['key'] for q in p['patches']} | set(p.get('verts', {}))
        self.proposal = None
        res = self.commit(p['label'], dele=p['del'], add=add, verts=p.get('verts'))
        self._touch(keys)
        return res

    def discard(self):
        if self.proposal:
            keys = set(self.proposal['del']) | {q['key'] for q in self.proposal['patches']} | set(self.proposal.get('verts', {}))
            self.proposal = None
            self._touch(keys)
        return dict(ok=True)

    # ---------------------------------------------------------------- export & road segmentation
    @staticmethod
    def _compute_normals(V, T):
        if not len(V) or not len(T):
            return np.zeros_like(V, dtype=np.float32)
        v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        tr = T.ravel()
        vn = np.stack([np.bincount(tr, np.repeat(fn[:, c], 3), minlength=len(V)) for c in range(3)], axis=1).astype(np.float32)
        norm = np.linalg.norm(vn, axis=1, keepdims=True)
        return np.where(norm > 1e-9, vn / np.maximum(norm, 1e-9), np.array([0., 0., 1.], dtype=np.float32))

    def export_road_segmented(self, prm=None):
        """Export raw edited mesh/cloud (with current studio edits applied) and/or Road-segmented labelled PLY
        (class Road = 1 inside lane boundaries, light colour with half transparency alpha = 128)."""
        prm = prm or {}
        t0 = time.time()
        arc0, arc1 = float(prm.get('arc0', 0)), float(prm.get('arc1', 1e9))
        margin = float(prm.get('margin', 0.0))
        hmin, hmax = float(prm.get('hmin', -0.8)), float(prm.get('hmax', 0.6))
        rgb = [int(x) for x in prm.get('rgb', [140, 210, 255])]
        raw_alpha = float(prm.get('alpha', 0.5))
        alpha = int(round(raw_alpha * 255)) if raw_alpha <= 1.0 else int(round(raw_alpha))
        alpha = int(np.clip(alpha, 0, 255))
        blend = float(np.clip(prm.get('blend', 0.0), 0.0, 1.0))
        save_raw = bool(prm.get('save_raw', True))
        save_labelled = bool(prm.get('save_labelled', True))
        save_road_only = bool(prm.get('save_road_only', True))
        save_zup = bool(prm.get('save_zup', True))
        save_parts = bool(prm.get('save_parts', False))

        VERT_RAW = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        FACE_RAW = np.dtype([('count', 'u1'), ('v0', '<u4'), ('v1', '<u4'), ('v2', '<u4')])
        VERT_LAB = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'), ('label', 'u1'), ('class_id', 'u1')])
        FACE_LAB = np.dtype([('count', 'u1'), ('v0', '<u4'), ('v1', '<u4'), ('v2', '<u4'), ('label', 'u1'), ('class_id', 'u1')])
        HEADER_PAD = 1024

        class _Writer:
            def __init__(self, path, labelled=False):
                self.path = path
                self.labelled = labelled
                self.v_tmp = path + '.verts.tmp'
                self.f_tmp = path + '.faces.tmp'
                self.fv = open(self.v_tmp, 'wb')
                self.ff = open(self.f_tmp, 'wb')
                self.fv.write(b' ' * HEADER_PAD)
                self.nv = 0
                self.nf = 0

            def write_chunk(self, V, N, C, T, v_road=None, f_road=None):
                if not len(V) or not len(T):
                    return
                if not self.labelled:
                    va = np.empty(len(V), dtype=VERT_RAW)
                    va['x'], va['y'], va['z'] = V[:, 0], V[:, 1], V[:, 2]
                    va['nx'], va['ny'], va['nz'] = N[:, 0], N[:, 1], N[:, 2]
                    va['red'], va['green'], va['blue'] = C[:, 0], C[:, 1], C[:, 2]
                    self.fv.write(va.tobytes())
                    fa = np.empty(len(T), dtype=FACE_RAW)
                    fa['count'] = 3
                    fa['v0'], fa['v1'], fa['v2'] = T[:, 0] + self.nv, T[:, 1] + self.nv, T[:, 2] + self.nv
                    self.ff.write(fa.tobytes())
                else:
                    va = np.empty(len(V), dtype=VERT_LAB)
                    va['x'], va['y'], va['z'] = V[:, 0], V[:, 1], V[:, 2]
                    va['nx'], va['ny'], va['nz'] = N[:, 0], N[:, 1], N[:, 2]
                    c_out = C.copy()
                    if v_road is not None and v_road.any():
                        rc = np.array(rgb, dtype=np.float32)
                        if blend > 0:
                            c_out[v_road] = np.clip(np.round(blend * C[v_road].astype(np.float32) + (1.0 - blend) * rc), 0, 255).astype(np.uint8)
                        else:
                            c_out[v_road] = np.array(rgb, dtype=np.uint8)
                    va['red'], va['green'], va['blue'] = c_out[:, 0], c_out[:, 1], c_out[:, 2]
                    va['alpha'] = np.where(v_road, alpha, 255).astype(np.uint8) if v_road is not None else 255
                    lab_v = v_road.astype(np.uint8) if v_road is not None else np.zeros(len(V), dtype=np.uint8)
                    va['label'] = lab_v
                    va['class_id'] = lab_v
                    self.fv.write(va.tobytes())
                    fa = np.empty(len(T), dtype=FACE_LAB)
                    fa['count'] = 3
                    fa['v0'], fa['v1'], fa['v2'] = T[:, 0] + self.nv, T[:, 1] + self.nv, T[:, 2] + self.nv
                    lab_f = f_road.astype(np.uint8) if f_road is not None else np.zeros(len(T), dtype=np.uint8)
                    fa['label'] = lab_f
                    fa['class_id'] = lab_f
                    self.ff.write(fa.tobytes())
                self.nv += len(V)
                self.nf += len(T)

            def close(self):
                self.ff.close()
                with open(self.f_tmp, 'rb') as rf:
                    shutil.copyfileobj(rf, self.fv, length=64 * 1024 * 1024)
                os.remove(self.f_tmp)
                lines = [
                    'ply',
                    'format binary_little_endian 1.0',
                    'comment Created by MeshSfM SplatStudio - Road Segmentation',
                ]
                if self.labelled:
                    lines += [
                        'comment classes: 0=Unlabelled, 1=Road',
                        'comment class 0 Unlabelled',
                        'comment class 1 Road',
                        f'comment road_rgba: {rgb[0]} {rgb[1]} {rgb[2]} {alpha}',
                    ]
                lines += [
                    f'element vertex {self.nv}',
                    'property float x', 'property float y', 'property float z',
                    'property float nx', 'property float ny', 'property float nz',
                    'property uchar red', 'property uchar green', 'property uchar blue',
                ]
                if self.labelled:
                    lines += ['property uchar alpha', 'property uchar label', 'property uchar class_id']
                lines += [
                    f'element face {self.nf}',
                    'property list uchar uint vertex_indices',
                ]
                if self.labelled:
                    lines += ['property uchar label', 'property uchar class_id']
                body = '\n'.join(lines) + '\n'
                end = 'end_header\n'
                pad_len = HEADER_PAD - len(body) - len(end) - len('comment ') - 1
                hdr = (body + 'comment ' + (' ' * pad_len) + '\n' + end).encode('ascii')
                self.fv.seek(0)
                self.fv.write(hdr)
                self.fv.close()
                os.replace(self.v_tmp, self.path)

        os.makedirs( os.path.dirname(ROUTE_OUT), exist_ok=True)
        writers = {}
        saved_files = []
        if save_raw:
            p_raw_s = ROUTE_OUT + '_raw_edited_source_frame.ply'
            writers['raw_splat'] = _Writer(p_raw_s, labelled=False); saved_files.append(p_raw_s)
            if save_zup:
                p_raw_z = ROUTE_OUT + '_raw_edited_zup.ply'
                writers['raw_zup'] = _Writer(p_raw_z, labelled=False); saved_files.append(p_raw_z)
        if save_labelled:
            p_lab_s = ROUTE_OUT + '_road_segmented_source_frame.ply'
            writers['lab_splat'] = _Writer(p_lab_s, labelled=True); saved_files.append(p_lab_s)
            if save_zup:
                p_lab_z = ROUTE_OUT + '_road_segmented_zup.ply'
                writers['lab_zup'] = _Writer(p_lab_z, labelled=True); saved_files.append(p_lab_z)
            if save_road_only:
                p_ro_s = ROUTE_OUT + '_road_only_source_frame.ply'
                writers['ro_splat'] = _Writer(p_ro_s, labelled=True); saved_files.append(p_ro_s)
                if save_zup:
                    p_ro_z = ROUTE_OUT + '_road_only_zup.ply'
                    writers['ro_zup'] = _Writer(p_ro_z, labelled=True); saved_files.append(p_ro_z)

        part_writers_raw, part_writers_lab = {}, {}
        if save_parts:
            os.makedirs(PARTS_OUT, exist_ok=True)
            parts_lab_dir = PARTS_OUT + '_road_segmented'
            os.makedirs(parts_lab_dir, exist_ok=True)
            for pi, name in enumerate(self.meta['parts'], 1):
                if save_raw:
                    part_writers_raw[pi] = _Writer(os.path.join(PARTS_OUT, name), labelled=False)
                if save_labelled:
                    part_writers_lab[pi] = _Writer(os.path.join(parts_lab_dir, name), labelled=True)

        road_v_list, road_n_list, road_t_list, road_v_off = [], [], [], 0
        total_tris, total_road_tris, total_verts, total_road_verts = 0, 0, 0, 0
        sorted_keys = sorted(self.chunks.keys())

        def process_piece(V_zup, C, T, arc, lat, H, pi):
            nonlocal road_v_off, total_tris, total_road_tris, total_verts, total_road_verts
            if not len(T):
                return
            used, inv = np.unique(T.ravel(), return_inverse=True)
            V_z = V_zup[used].astype(np.float32)
            C_u = C[used].astype(np.uint8)
            T_u = inv.reshape(-1, 3).astype(np.uint32)
            N_z = self._compute_normals(V_z, T_u)
            V_s = to_source(V_z).astype(np.float32)
            N_s = to_source_dir(N_z).astype(np.float32)

            lo, hi = self.edges_at(arc)
            f_road = (arc >= arc0) & (arc <= arc1) & (lat >= lo - margin) & (lat <= hi + margin) & (H >= hmin) & (H <= hmax)
            v_road = np.zeros(len(V_z), dtype=bool)
            if f_road.any():
                v_road[T_u[f_road].ravel()] = True

            total_verts += int(len(V_z))
            total_road_verts += int(v_road.sum())
            total_tris += int(len(T_u))
            total_road_tris += int(f_road.sum())

            if 'raw_splat' in writers:
                writers['raw_splat'].write_chunk(V_s, N_s, C_u, T_u)
            if 'raw_zup' in writers:
                writers['raw_zup'].write_chunk(V_z, N_z, C_u, T_u)
            if 'lab_splat' in writers:
                writers['lab_splat'].write_chunk(V_s, N_s, C_u, T_u, v_road, f_road)
            if 'lab_zup' in writers:
                writers['lab_zup'].write_chunk(V_z, N_z, C_u, T_u, v_road, f_road)
            if pi in part_writers_raw:
                part_writers_raw[pi].write_chunk(V_s, N_s, C_u, T_u)
            if pi in part_writers_lab:
                part_writers_lab[pi].write_chunk(V_s, N_s, C_u, T_u, v_road, f_road)

            if f_road.any() and (save_labelled or 'ro_splat' in writers):
                Tr = T_u[f_road]
                r_used, r_inv = np.unique(Tr.ravel(), return_inverse=True)
                Vr_s, Nr_s, Cr = V_s[r_used], N_s[r_used], C_u[r_used]
                Vr_z, Nr_z = V_z[r_used], N_z[r_used]
                Tr_u = r_inv.reshape(-1, 3).astype(np.uint32)
                vr_ones = np.ones(len(Vr_s), dtype=bool)
                fr_ones = np.ones(len(Tr_u), dtype=bool)
                if 'ro_splat' in writers:
                    writers['ro_splat'].write_chunk(Vr_s, Nr_s, Cr, Tr_u, vr_ones, fr_ones)
                if 'ro_zup' in writers:
                    writers['ro_zup'].write_chunk(Vr_z, Nr_z, Cr, Tr_u, vr_ones, fr_ones)
                if save_labelled:
                    road_v_list.append(Vr_s)
                    road_n_list.append(Nr_s)
                    road_t_list.append(Tr_u + road_v_off)
                    road_v_off += len(Vr_s)

        for idx_k, k in enumerate(sorted_keys):
            self._progress(f'segmenting & writing {k}', idx_k / max(len(sorted_keys), 1) * 0.9)
            c = self.chunks[k]
            pi = c['part']
            d = self.load(k)
            keep = ~self.dmask(k)
            if keep.any():
                Tk = np.asarray(d['T'])[keep]
                Vk, Ck = self.verts(k)
                Vk_abs = Vk.astype(np.float64) + self.O
                arc = np.asarray(d['arc'])[keep]
                lat = np.asarray(d['lat'])[keep]
                H = np.asarray(self.Hcur(k))[keep] / 100.0
                process_piece(Vk_abs, Ck, Tk, arc, lat, H, pi)
            for pid, p in self.patches.items():
                if p['active'] and p['key'] == k and len(p['T']):
                    Vp_abs = p['V'].astype(np.float64) + self.O
                    Tp = np.asarray(p['T'])
                    cen = Vp_abs[Tp].mean(1)
                    arc_p, lat_p = self.route.project(cen[:, :2])
                    H_p = cen[:, 2] - self.ground_at(cen[:, :2])
                    process_piece(Vp_abs, p['C'], Tp, arc_p, lat_p, H_p, pi)

        self._progress('finalising PLY headers & label index...', 0.93)
        for w in writers.values():
            w.close()
        for w in part_writers_raw.values():
            w.close()
        for w in part_writers_lab.values():
            w.close()

        npz_path = None
        if save_labelled and road_v_list:
            npz_path = ROUTE_OUT + '_road_labels_source_frame.npz'
            s_c, lo_c, hi_c = self.corridor()
            np.savez(
                npz_path,
                road_vertices_splat=np.concatenate(road_v_list),
                road_normals_splat=np.concatenate(road_n_list),
                road_triangles=np.concatenate(road_t_list),
                corridor_arc=s_c.astype(np.float32),
                corridor_left_lat=lo_c.astype(np.float32),
                corridor_right_lat=hi_c.astype(np.float32),
                class_id=np.uint8(1),
                class_name=np.array('Road'),
                road_rgba=np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.uint8),
                margin=np.float32(margin),
                hmin=np.float32(hmin),
                hmax=np.float32(hmax),
            )
            saved_files.append(npz_path)

        dt = time.time() - t0
        return dict(
            ok=True,
            road_seg_export=True,
            triangles=total_tris,
            road_triangles=total_road_tris,
            vertices=total_verts,
            road_vertices=total_road_verts,
            files=saved_files,
            npz=npz_path,
            seconds=round(dt, 1),
        )
    def export(self, write_obj=False):
        import open3d as o3d
        os.makedirs(PARTS_OUT, exist_ok=True)
        parts = self.meta['parts']
        allV, allT, allC, off = [], [], [], 0
        for pi, name in enumerate(parts, 1):
            self._progress(f'export {name}', (pi - 1) / len(parts))
            Vs, Ts, Cs, o = [], [], [], 0
            for k, c in sorted(self.chunks.items()):
                if c['part'] != pi:
                    continue
                d = self.load(k)
                Tk = np.asarray(d['T'])[~self.dmask(k)]
                Vk, Ck = self.verts(k)
                Vs.append(Vk); Cs.append(Ck); Ts.append(Tk.astype(np.int64) + o); o += len(Vk)
            for pid, p in self.patches.items():
                if p['active'] and self.chunks[p['key']]['part'] == pi:
                    Vs.append(p['V']); Cs.append(p['C']); Ts.append(p['T'].astype(np.int64) + o); o += len(p['V'])
            V = np.concatenate(Vs).astype(np.float64) + self.O
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.concatenate(Ts)))
            m.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(Cs) / 255.0)
            m.merge_close_vertices(0.002); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
            o3d.io.write_triangle_mesh(os.path.join(PARTS_OUT, name), m, write_vertex_normals=True)
            allV.append(np.asarray(m.vertices)); allT.append(np.asarray(m.triangles) + off); allC.append(np.asarray(m.vertex_colors)); off += len(m.vertices)
        self._progress('export merged route mesh', 0.95)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(allV)), o3d.utility.Vector3iVector(np.concatenate(allT)))
        m.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(allC))
        m.merge_close_vertices(0.02); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
        o3d.io.write_triangle_mesh(ROUTE_OUT + '_zup.ply', m, write_vertex_normals=True)
        if write_obj:
            o3d.io.write_triangle_mesh(ROUTE_OUT + '_zup.obj', m, write_vertex_normals=True)
        Ts_, fname = PROJECT.source_frame()
        if not np.allclose(Ts_, np.eye(4)):        # also in the frame of the source mesh
            m.vertices = o3d.utility.Vector3dVector(to_source(np.asarray(m.vertices))); m.compute_vertex_normals()
            o3d.io.write_triangle_mesh(ROUTE_OUT + '_source_frame.ply', m, write_vertex_normals=True)
        return dict(parts=PARTS_OUT, route=ROUTE_OUT + '_*.ply', triangles=int(len(m.triangles)))


# ======================================================================= lanes (annotation export)
def catmull_rom(nodes, smooth=True, step=0.05):
    P = np.asarray(nodes, float)
    if len(P) < 2:
        return P
    if not smooth or len(P) == 2:
        out = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)))
            out.extend(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
        return np.array(out)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = [P[0]]
    for s in range(len(P) - 1):
        p0, p1, p2, p3 = ext[s], ext[s + 1], ext[s + 2], ext[s + 3]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        n = max(2, int(math.ceil(np.linalg.norm(p2 - p1) / step)))
        t = (t1 + (t2 - t1) * (np.arange(1, n + 1) / n))[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.extend((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.array(out)


def resample(xz, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    s = np.arange(0, d[-1] + 1e-9, step)
    if d[-1] - s[-1] > 1e-3:
        s = np.r_[s, d[-1]]
    return np.c_[np.interp(s, d, xz[:, 0]), np.interp(s, d, xz[:, 1])]


def douglas_peucker(P, tol):
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    keep = np.zeros(len(P), bool); keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = P[b] - P[a]; L = np.linalg.norm(ab) + 1e-12
        d = np.abs(np.cross(ab, P[a + 1:b] - P[a])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return P[keep]


class Surface:
    """Ground heights (z-up) at (x, y) from the ground-surface points, guided by the path height."""
    def __init__(self, route):
        from scipy.spatial import cKDTree
        self.route = route
        self.P, self.arc = route.P, route.arc
        self.rtree = cKDTree(self.P[:, :2])
        self.surface = None
        self.lock = threading.Lock()

    def surf(self):
        from scipy.spatial import cKDTree
        with self.lock:
            if self.surface is None:
                S = np.load(os.path.join(TILES, 'surface_xyz.npy'))
                self.surface = (S, cKDTree(S[:, :2]))
        return self.surface

    def heights(self, xy, smooth=4):
        from scipy.ndimage import gaussian_filter1d
        xy = np.asarray(xy, float).reshape(-1, 2)
        S, tr = self.surf()
        _, j = self.rtree.query(xy)
        zr = self.P[j, 2]
        lat = np.linalg.norm(xy - self.P[j, :2], axis=1)
        tol = 0.5 + 0.1 * np.maximum(0, lat - 2.5)
        z = np.full(len(xy), np.nan)
        for i, idx in enumerate(tr.query_ball_point(xy, 0.3)):
            if len(idx) >= 3:
                v = S[idx, 2]; v = v[np.abs(v - zr[i]) < tol[i]]
                if len(v) >= 3:
                    z[i] = np.median(v)
        ok = np.isfinite(z)
        if not ok.any():
            return zr
        z = np.interp(np.arange(len(z)), np.flatnonzero(ok), z[ok])
        return gaussian_filter1d(z, smooth, mode='nearest') if smooth else z

    def arc_of(self, xy):
        return self.route.project(xy)[0]


def export_lanes(surface, ann):
    os.makedirs(ANN, exist_ok=True)
    colours = {'left_edge': (255, 0, 0), 'right_edge': (0, 80, 255), 'centre_line': (255, 220, 0),
               'lane_divider': (255, 255, 255), 'stop_line': (255, 140, 0), 'other': (255, 0, 255)}
    rows, pts, rgb, per_type = [], [], [], {}
    for ln in ann['lines']:
        if len(ln['nodes']) < 2:
            continue
        xy = resample(catmull_rom(ln['nodes'], ln.get('smooth', True)), 0.25)
        X = np.c_[xy, surface.heights(xy)]
        rows += [(ln['id'], ln['type'], k, *x) for k, x in enumerate(X)]
        pts.append(X); rgb.append(np.tile(colours.get(ln['type'], (200, 200, 200)), (len(X), 1)))
        per_type.setdefault(ln['type'], []).append(X)
    for frame, F in [('zup', None), ('source_frame', to_source)]:
        with open(os.path.join(ANN, f'lines_{frame}.csv'), 'w') as f:
            f.write('line_id,type,index,x,y,z\n')
            for r in rows:
                x = np.array(r[3:]) if F is None else F(np.array(r[3:])[None])[0]
                f.write(f'{r[0]},{r[1]},{r[2]},{x[0]:.4f},{x[1]:.4f},{x[2]:.4f}\n')
    if pts:
        pts, rgb = np.concatenate(pts), np.concatenate(rgb).astype(np.uint8)
        v = np.empty(len(pts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        v['x'], v['y'], v['z'] = pts.T
        v['red'], v['green'], v['blue'] = rgb.T
        with open(os.path.join(ANN, 'lines_source_frame.ply'), 'wb') as f:
            f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n'
                     'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % len(v)).encode())
            f.write(v.tobytes())
    st = np.arange(0, surface.arc[-1], 1.0)

    def along(kind):
        if kind not in per_type:
            return np.full((len(st), 3), np.nan), np.zeros(len(st), bool)
        X = np.concatenate(per_type[kind])
        a = surface.arc_of(X[:, :2])
        o = np.argsort(a); a, X = a[o], X[o]
        a, u = np.unique(a, return_index=True); X = X[u]
        out = np.stack([np.interp(st, a, X[:, k]) for k in range(3)], 1)
        i = np.clip(np.searchsorted(a, st), 0, len(a) - 1)
        near = np.minimum(np.abs(a[i] - st), np.abs(a[np.clip(i - 1, 0, len(a) - 1)] - st))
        have = near < 1.0
        out[~have] = np.nan
        return out, have
    left, hl = along('left_edge'); right, hr = along('right_edge'); centre, hc = along('centre_line')
    centre = np.where(hc[:, None], centre, 0.5 * (left + right))
    ll, rl = 0.5 * (left + centre), 0.5 * (right + centre)
    width = np.linalg.norm(right - left, axis=1)
    gap = ~(hl & hr); long_gap = np.zeros(len(st), bool); k = 0
    while k < len(st):
        if gap[k]:
            e = k
            while e < len(st) and gap[e]:
                e += 1
            long_gap[k:e] = (e - k) > 40; k = e
        else:
            k += 1
    cols, data = ['arc_m'], [st]
    for name, arr in [('left', left), ('right', right), ('centre', centre), ('left_lane_centre', ll), ('right_lane_centre', rl)]:
        cols += [f'{name}_x', f'{name}_y', f'{name}_z']; data += [arr[:, 0], arr[:, 1], arr[:, 2]]
    cols += ['width', 'left_seen', 'right_seen', 'long_gap', 'confidence']
    data += [width, hl.astype(int), hr.astype(int), long_gap.astype(int), (hl & hr).astype(float)]
    table = np.stack(data, 1)
    for frame, F in [('zup', None), ('source_frame', to_source)]:
        tt = table.copy()
        for k in range(1, 16, 3):
            if F is not None:
                tt[:, k:k + 3] = F(tt[:, k:k + 3])
        np.savetxt(os.path.join(ANN, f'lanes_annotated_{frame}.csv'), tt, delimiter=',', header=','.join(cols),
                   comments='', fmt=['%.2f'] + ['%.4f'] * 15 + ['%.3f', '%d', '%d', '%d', '%.0f'])
        json.dump({'frame': frame, 'units': 'metres', 'columns': cols,
                   'rows': [[None if not np.isfinite(v) else round(float(v), 4) for v in r] for r in tt]},
                  open(os.path.join(ANN, f'lanes_annotated_{frame}.json'), 'w'))
    return dict(lines=len(ann['lines']), both_edges_m=int((hl & hr).sum()), left_m=int(hl.sum()), right_m=int(hr.sum()))
