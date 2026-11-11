"""Align one layer to another (splat <-> mesh <-> points, any combination).

Point pairs picked by hand are treated as noisy anchors, not as the truth:

  1. pairs -> start: RANSAC over minimal subsets throws out mis-clicked pairs, then a least-squares fit
     (translation from 1 pair, yaw + translation from 2, rigid / similarity from 3+)
  2. several starts (pairs, current placement, optional global FPFH match) are scored after a coarse pass; the best goes on
  3. one robust Gauss-Newton solve, coarse to fine (voxels 0.5 m -> 3 cm):
       point-to-plane residuals (Geman-McClure weights, scale re-estimated every iteration)
     + photometric residuals on the fixed layer's colour gradient (gain / offset fitted, so mesh vs splat exposure does not matter)
     + the pairs as soft anchors (Cauchy weights), which pin directions the geometry leaves free (a straight flat road)
     with a chosen number of degrees of freedom: rigid, similarity, yaw (keeps up = up), translation
  4. report: overlap / gap before and after, residual of every pair (mis-clicks flagged), uncertainty per axis and
     the weakest direction, so it is clear when more pairs are needed and where.

The result is a correction D (4x4, internal frame): the moving layer transform becomes D @ T.
"""
import itertools
import numpy as np

from .project import Project, apply

MODES = ('rigid', 'similarity', 'yaw', 'translation')
DOF = dict(rigid=[0, 1, 2, 3, 4, 5], similarity=[0, 1, 2, 3, 4, 5, 6], yaw=[2, 3, 4, 5], translation=[3, 4, 5])   # (wx wy wz tx ty tz s)


def log(msg):
    print(msg, flush=True)


def layer_points(P, L, n=1_500_000):
    from . import prep
    T = P.T(L)
    if L['kind'] == 'mesh':
        X, C = [], []
        for f in P.files_of(L):
            V, Cv, _ = prep.read_mesh(f, T); X.append(V); C.append(Cv)
        X, C = np.concatenate(X), np.concatenate(C) / 255.0
    elif L['kind'] == 'points':
        out = [prep.read_points(f, T) for f in P.files_of(L)]
        X, C = np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]) / 255.0
    else:
        X, C, op, sc, _ = prep.read_splat(L['src'], T, max_n=4 * n)
        k = (op > 0.4) & (sc.max(1) < np.log(0.3)); X, C = X[k], C[k]
    if len(X) > n:
        s = np.random.default_rng(0).choice(len(X), n, replace=False); X, C = X[s], C[s]
    return X, C


# ------------------------------------------------------------------------------------------------ closed-form fits
def umeyama(A, B, scale=True, w=None):
    """B ~ s R A + t (weighted)."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float); w = w / w.sum()
    ma, mb = w @ A, w @ B; a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd((b * w[:, None]).T @ a); D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (w @ (a ** 2).sum(1)) if scale else 1.0
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = mb - s * R @ ma
    return T


def fit_pairs(A, B, mode):
    """Least-squares transform of the given mode from pairs A -> B (degrades gracefully with few pairs)."""
    n = len(A); T = np.eye(4)
    if n == 0:
        return T
    if mode == 'translation' or n == 1:
        T[:3, 3] = (B - A).mean(0); return T
    if mode in ('yaw', 'yawscale') or n == 2:
        a, b = A[:, :2] - A[:, :2].mean(0), B[:, :2] - B[:, :2].mean(0)
        th = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
        s = 1.0
        if mode in ('similarity', 'yawscale'):
            s = np.sqrt((b ** 2).sum() / max((a ** 2).sum(), 1e-12))
        c, si = np.cos(th), np.sin(th); R = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1]]) * s
        T[:3, :3] = R; T[:3, 3] = B.mean(0) - R @ A.mean(0); return T
    return umeyama(A, B, scale=(mode == 'similarity'))


def robust_pairs(A, B, mode, tol):
    """RANSAC over minimal subsets (all of them for a handful of pairs): the largest consistent set, refit."""
    n = len(A); m = dict(translation=1, yaw=2, yawscale=2).get(mode, 3)
    if n <= m:
        return fit_pairs(A, B, mode), np.ones(n, bool)
    subsets = list(itertools.combinations(range(n), m))
    if len(subsets) > 400:
        rng = np.random.default_rng(0); subsets = [tuple(rng.choice(n, m, replace=False)) for _ in range(400)]
    best, best_key = None, None
    for s in subsets:
        T = fit_pairs(A[list(s)], B[list(s)], mode)
        r = np.linalg.norm(apply(T, A) - B, axis=1); inl = r < tol
        key = (inl.sum(), -np.sum(np.minimum(r, tol) ** 2))
        if best_key is None or key > best_key:
            best_key, best = key, inl
    if best.sum() < m:
        best = np.ones(n, bool)
    for _ in range(3):                                            # refit on the inliers, re-check
        T = fit_pairs(A[best], B[best], mode)
        r = np.linalg.norm(apply(T, A) - B, axis=1); nb = r < tol
        if nb.sum() < m or (nb == best).all():
            break
        best = nb
    return fit_pairs(A[best], B[best], mode), best


# ------------------------------------------------------------------------------------------------ helpers
def _pcd(X, v=None, normals_r=None):
    import open3d as o3d
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    if v:
        p = p.voxel_down_sample(v)
    if normals_r:
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normals_r, max_nn=30))
    return p


def voxel_down(X, C, v):
    """Voxel centroids of points + colours."""
    k = np.floor(X / v).astype(np.int64); k -= k.min(0)
    key = (k[:, 0] * (k[:, 1].max() + 1) + k[:, 1]) * (k[:, 2].max() + 1) + k[:, 2]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    Xo = np.zeros((len(u), 3)); Co = np.zeros((len(u), 3))
    np.add.at(Xo, inv, X); np.add.at(Co, inv, C)
    return Xo / cnt[:, None], Co / cnt[:, None]


def intensity(C):
    return C @ np.array([0.299, 0.587, 0.114])


def colour_gradients(Xf, Nf, If, tree, idx, r):
    """Gradient of intensity in the tangent plane of fixed points idx (Park et al. 2017 coloured ICP)."""
    d, nb = tree.query(Xf[idx], k=12, distance_upper_bound=r, workers=-1)
    ok = np.isfinite(d); nb = np.where(ok, nb, 0)
    q, n = Xf[idx], Nf[idx]
    A = Xf[nb] - q[:, None]; A -= (A @ n[:, :, None]) * n[:, None]     # neighbours projected onto the tangent plane
    b = If[nb] - If[idx][:, None]
    A *= ok[..., None]; b *= ok
    M = np.einsum('nki,nkj->nij', A, A) + 1e3 * np.einsum('ni,nj->nij', n, n) + 1e-6 * np.eye(3)
    g = np.linalg.solve(M, np.einsum('nki,nk->ni', A, b)[..., None])[..., 0]
    g -= (g * n).sum(1, keepdims=True) * n
    g[ok.sum(1) < 4] = 0
    return g


def increment(xi, c):
    """xi = (wx wy wz tx ty tz s) about pivot c -> 4x4."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(xi[:3]).as_matrix() * (1 + xi[6])
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = c + xi[3:6] - R @ c
    return E


def jac(Y, c):
    """d(point)/d(xi) for points Y (already transformed), pivot c: (n, 3, 7)."""
    p = Y - c; n = len(Y); J = np.zeros((n, 3, 7))
    J[:, 0, 1], J[:, 0, 2] = p[:, 2], -p[:, 1]                 # -[p]x
    J[:, 1, 0], J[:, 1, 2] = -p[:, 2], p[:, 0]
    J[:, 2, 0], J[:, 2, 1] = p[:, 1], -p[:, 0]
    J[:, 0, 3] = J[:, 1, 4] = J[:, 2, 5] = 1
    J[:, :, 6] = p
    return J


def rot_deg(D):
    R = D[:3, :3] / np.cbrt(abs(np.linalg.det(D[:3, :3])))
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ------------------------------------------------------------------------------------------------ the solver
def register(Xm, Cm, Xf, Cf, picks=None, mode='rigid', max_dist=1.0, pick_sigma=0.2, colour=0.3, global_search='auto',
             fine=0.03, D0=None):
    """Robust registration of moving points onto fixed points. Returns (D, report)."""
    from scipy.spatial import cKDTree
    mode = mode if mode in MODES else 'rigid'; dof = DOF[mode]
    picks = np.asarray(picks if picks is not None else [], float).reshape(-1, 6)
    PA, PB = picks[:, :3], picks[:, 3:]
    rep = dict(mode=mode, pairs=[], stages=[], hypotheses=[])
    D_start = np.eye(4) if D0 is None else np.asarray(D0, float)

    # ---- 1. starts
    # The start from the pairs keeps up = up (heading + shift, + scale): 2 pairs determine it, so 3 pairs can already vote out a
    # mis-click and nearly collinear pairs cannot flip the layer over. Tilt is left to the surfaces; a full 3D fit from the
    # pairs is only an extra candidate when there are enough well spread pairs and it agrees.
    starts = []
    use = np.ones(len(PA), bool); tol = max(4 * pick_sigma, 0.5)
    if len(PA):
        smode = dict(translation='translation', similarity='yawscale').get(mode, 'yaw')
        A0 = apply(D_start, PA)
        Dp, use = robust_pairs(A0, PB, smode, tol); Dp = Dp @ D_start
        r = np.linalg.norm(apply(Dp, PA) - PB, axis=1)
        log(f'pairs: {len(PA)} given, {use.sum()} consistent (tol {tol:.2f} m), RMS {np.sqrt((r[use] ** 2).mean()):.3f} m')
        for i in np.where(~use)[0]:
            log(f'  pair {i + 1} ignored as a start: off by {r[i]:.2f} m')
        starts.append(('pairs', Dp))
        if mode in ('rigid', 'similarity') and use.sum() >= 4:
            c = A0[use] - A0[use].mean(0); sv = np.linalg.svd(c, compute_uv=False)
            if sv[2] > 0.15 * sv[0]:                                     # spread in 3D, not on a line / in a plane
                Df = fit_pairs(A0[use], PB[use], mode) @ D_start
                if rot_deg(Df @ np.linalg.inv(Dp)) < 15:
                    starts.append(('pairs, full 3D', Df))
    starts.append(('current placement', D_start))

    # coarse working sets
    vc = max(0.5, max_dist / 2)
    Fc, FCc = voxel_down(Xf, Cf, vc / 2)
    tf_c = cKDTree(Fc)

    def overlap(D, X, gate):
        d, _ = tf_c.query(apply(D, X), distance_upper_bound=gate, workers=-1)
        ok = np.isfinite(d); return ok.mean(), (np.median(d[ok]) if ok.any() else np.inf)

    Mc, MCc = voxel_down(Xm, Cm, vc / 2)
    if global_search is True or (global_search == 'auto' and not len(PA) and overlap(D_start, Mc, 2 * max_dist)[0] < 0.3):
        Dg = global_match(apply(D_start, Mc), Fc, vc)
        if Dg is not None:
            starts.append(('global FPFH match', Dg @ D_start))

    # ---- 2. score the starts after a short coarse pass
    best = None
    for name, D in starts:
        D1, _ = solve(D, Xm, Cm, Xf, Cf, PA[use], PB[use], dof, voxel=vc, gate=max(2 * max_dist, 3 * vc), iters=15,
                      pick_sigma=pick_sigma, colour=0.0)
        ov, med = overlap(D1, Mc, vc)
        score = ov - 0.2 * min(med / vc, 1)
        if len(PA):                                                      # and the pairs should agree with it
            score -= 0.6 * np.mean(np.linalg.norm(apply(D1, PA[use]) - PB[use], axis=1) > tol)
        rep['hypotheses'].append(dict(start=name, overlap=float(ov), median=float(med), score=float(score)))
        log(f'start "{name}": overlap {ov:.1%}, median {med:.3f} m after coarse pass')
        if best is None or score > best[0]:
            best = (score, name, D1)
    _, chosen, D = best; rep['chosen'] = chosen

    # ---- 3. coarse to fine
    levels = [v for v in (0.5, 0.25, 0.12, 0.06, 0.03, 0.02) if v <= max(max_dist, 0.5) and v >= fine * 0.99]
    for i, v in enumerate(levels):
        gate = max(max_dist, 3 * v) if i == 0 else 3 * v + 0.02
        D, info = solve(D, Xm, Cm, Xf, Cf, PA, PB, dof, voxel=v, gate=gate, iters=30, pick_sigma=pick_sigma,
                        colour=colour if v <= 0.12 else 0.0)
        info.update(voxel=v); rep['stages'].append(info)
        log(f'  voxel {v:.2f} m: {info["iters"]} it, {info["n"]:,} matches, geometric sigma {100 * info["sigma"]:.1f} cm'
            + (f', colour {info["colour_w"]:.2f}' if info.get('colour_w') else ''))
    rep['uncertainty'] = rep['stages'][-1].get('uncertainty') if rep['stages'] else None
    for st in rep['stages']:
        st.pop('uncertainty', None)

    # ---- 4. pairs after the solve
    if len(PA):
        r = np.linalg.norm(apply(D, PA) - PB, axis=1)
        for i, ri in enumerate(r):
            rep['pairs'].append(dict(i=i + 1, residual=float(ri), used_as_start=bool(use[i]), suspect=bool(ri > max(3 * pick_sigma, 0.3))))
        log('pair residuals: ' + ', '.join(f'{i + 1}: {100 * ri:.0f} cm' + (' (suspect)' if ri > max(3 * pick_sigma, 0.3) else '') for i, ri in enumerate(r)))
    return D, rep


def solve(D, Xm, Cm, Xf, Cf, PA, PB, dof, voxel, gate, iters=30, pick_sigma=0.2, colour=0.3, n_eff=3000, max_moving=250_000):
    """Robust Gauss-Newton at one resolution: point-to-plane (+ colour) + pair anchors."""
    from scipy.spatial import cKDTree
    # fixed: voxel centroids near the moving layer, normals; moving: voxel centroids, subsampled
    M, MC = voxel_down(Xm, Cm, voxel)
    if len(M) > max_moving:
        s = np.random.default_rng(1).choice(len(M), max_moving, replace=False); M, MC = M[s], MC[s]
    Y0 = apply(D, M)
    lo, hi = Y0.min(0) - gate - 1, Y0.max(0) + gate + 1
    k = ((Xf >= lo) & (Xf <= hi)).all(1)
    F, FC = voxel_down(Xf[k], Cf[k], voxel)
    info = dict(iters=0, n=0, sigma=float('nan'))
    if len(F) < 50 or len(M) < 50:
        return D, info
    tf = cKDTree(F)
    d0, _ = cKDTree(Y0).query(F, distance_upper_bound=gate + voxel, workers=-1)
    F, FC = F[np.isfinite(d0)], FC[np.isfinite(d0)]                     # only the overlap region of the fixed layer
    if len(F) < 50:
        return D, info
    tf = cKDTree(F)
    pc = _pcd(F, normals_r=max(2.5 * voxel, 0.1)); NF = np.asarray(pc.normals)
    IF, IM = intensity(FC), intensity(MC)
    have = np.zeros(len(F), bool); GR = np.zeros((len(F), 3)); H = Hg = None
    c = Y0.mean(0)
    sigma = gate / 3; wcol = 0.0
    for it in range(iters):
        Y = apply(D, M)
        d, j = tf.query(Y, distance_upper_bound=gate, workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        Yk, jk = Y[ok], j[ok]
        n = NF[jk]; r = ((Yk - F[jk]) * n).sum(1)
        sigma = max(1.4826 * np.median(np.abs(r)), 0.15 * voxel)
        e = r / sigma; w = 1 / (1 + e ** 2) ** 2                            # Geman-McClure
        Jp = jac(Yk, c); Jg = np.einsum('ni,nij->nj', n, Jp)
        scale = n_eff / len(r)                                              # neighbouring matches are not independent
        H = scale * (Jg * (w / sigma ** 2)[:, None]).T @ Jg; g = scale * (Jg * (w * r / sigma ** 2)[:, None]).T.sum(1)
        Hg = H.copy()
        if colour > 0:
            miss = np.unique(jk[~have[jk]])
            if len(miss):
                GR[miss] = colour_gradients(F, NF, IF, tf, miss, 3 * voxel); have[miss] = True
            G = GR[jk]
            Im = IM[ok]
            A = np.c_[Im, np.ones_like(Im)]; gain = np.linalg.lstsq(A * w[:, None], IF[jk] * w, rcond=None)[0]   # exposure
            rc = IF[jk] + (G * (Yk - F[jk] - (((Yk - F[jk]) * n).sum(1))[:, None] * n)).sum(1) - A @ gain
            sc = max(1.4826 * np.median(np.abs(rc)), 1e-3)
            ec = rc / sc; wc = colour / (1 + ec ** 2) ** 2 * w
            Jc = np.einsum('ni,nij->nj', G, Jp)
            H += scale * (Jc * (wc / sc ** 2)[:, None]).T @ Jc; g += scale * (Jc * (wc * rc / sc ** 2)[:, None]).T.sum(1)
            wcol = float(colour)
        if len(PA):                                                         # pair anchors (Cauchy: a mis-click fades out)
            Ya = apply(D, PA); ra = Ya - PB; na = np.linalg.norm(ra, axis=1)
            wa = 1 / (1 + (na / (2 * pick_sigma)) ** 2) / pick_sigma ** 2
            Ja = jac(Ya, c)
            for a in range(len(PA)):
                H += wa[a] * Ja[a].T @ Ja[a]; g += wa[a] * Ja[a].T @ ra[a]
        Hs, gs = H[np.ix_(dof, dof)], g[dof]
        Hs = Hs + 1e-9 * np.trace(Hs) * np.eye(len(dof))
        xs = -np.linalg.solve(Hs, gs); xi = np.zeros(7); xi[dof] = xs
        D = increment(xi, c) @ D
        info.update(iters=it + 1, n=int(ok.sum()), sigma=float(sigma))
        if np.abs(xi[:3]).max() < 1e-6 and np.abs(xi[3:6]).max() < 1e-4 and abs(xi[6]) < 1e-6:
            break
    info['colour_w'] = wcol
    if H is not None:
        info['uncertainty'] = uncertainty(Hg[np.ix_(dof, dof)], H[np.ix_(dof, dof)], dof)
    return D, info


def uncertainty(H_geo, H_all, dof):
    """1-sigma per axis from the geometric information alone and with the pairs; weakest geometric direction."""
    names = ['rot x', 'rot y', 'rot z', 'x', 'y', 'z', 'scale']

    def std(H):
        try:
            C = np.linalg.inv(H + 1e-12 * np.eye(len(H)))
        except np.linalg.LinAlgError:
            return None
        return {names[d]: float(np.sqrt(max(C[i, i], 0)) * (np.degrees(1) if d < 3 else 1)) for i, d in enumerate(dof)}
    out = dict(geometry=std(H_geo), with_pairs=std(H_all) if H_all is not None else None)
    t = [i for i, d in enumerate(dof) if 3 <= d < 6]
    if len(t) == 3:
        Ht = H_geo[np.ix_(t, t)]; ev, V = np.linalg.eigh(Ht)
        out['weakest_translation'] = dict(direction=V[:, 0].round(3).tolist(), sigma=float(1 / np.sqrt(max(ev[0], 1e-30))),
                                          ratio=float(ev[0] / max(ev[-1], 1e-30)))
    return out


def global_match(Xm, Xf, v):
    """FPFH + RANSAC (when there are no pairs and the layers barely overlap)."""
    import open3d as o3d
    reg = o3d.pipelines.registration

    def feat(X):
        p = _pcd(X, v, normals_r=2 * v)
        return p, reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * v, max_nn=100))
    try:
        (pm, fm), (pf, ff) = feat(Xm), feat(Xf)
        r = reg.registration_ransac_based_on_feature_matching(
            pm, pf, fm, ff, True, 1.5 * v, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(1.5 * v)],
            reg.RANSACConvergenceCriteria(200000, 0.999))
        log(f'global match: fitness {r.fitness:.3f}')
        return np.array(r.transformation) if r.fitness > 0.05 else None
    except Exception as e:
        log(f'global match failed: {e!r}'); return None


# ------------------------------------------------------------------------------------------------ studio entry points
def stats(D, Xm, Xf):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(Xf).query(apply(D, Xm[::max(1, len(Xm) // 200000)]), distance_upper_bound=1.0, workers=-1)
    ok = np.isfinite(d)
    return dict(overlap=float(ok.mean()), median=float(np.median(d[ok])) if ok.any() else None, within_5cm=float((d < 0.05).mean()),
                within_10cm=float((d < 0.10).mean()))


def run(project, body):
    """body: {moving, fixed, picks: [[mx,my,mz,fx,fy,fz]...], mode: rigid|similarity|yaw|translation (or scale: bool),
              icp: bool, max_dist: m, pick_sigma: m, colour: 0..1, global: auto|true|false, region: [x0,y0,x1,y1]}"""
    P = project if isinstance(project, Project) else Project(project)
    Lm, Lf = P.layer(body['moving']), P.layer(body['fixed'])
    Xm, Cm = layer_points(P, Lm); Xf, Cf = layer_points(P, Lf)
    log(f'align {Lm["name"]} -> {Lf["name"]}: {len(Xm):,} / {len(Xf):,} points')
    mode = body.get('mode') or ('similarity' if body.get('scale') else 'rigid')
    picks = np.asarray(body.get('picks') or [], float).reshape(-1, 6)
    if body.get('region'):
        x0, y0, x1, y1 = body['region']
        km = (Xm[:, 0] >= x0) & (Xm[:, 0] <= x1) & (Xm[:, 1] >= y0) & (Xm[:, 1] <= y1)
        kf = (Xf[:, 0] >= x0) & (Xf[:, 0] <= x1) & (Xf[:, 1] >= y0) & (Xf[:, 1] <= y1)
        Xm, Cm, Xf, Cf = Xm[km], Cm[km], Xf[kf], Cf[kf]
    before = stats(np.eye(4), Xm, Xf)
    md, ps = float(body.get('max_dist', 1.0)), float(body.get('pick_sigma', 0.2))
    if body.get('icp', True):
        g = body.get('global', 'auto'); g = g if g == 'auto' else bool(g)
        D, rep = register(Xm, Cm, Xf, Cf, picks, mode=mode, max_dist=md, pick_sigma=ps, colour=float(body.get('colour', 0.3)), global_search=g)
    else:
        D, use = robust_pairs(picks[:, :3], picks[:, 3:], mode, max(4 * ps, 0.5)) if len(picks) else (np.eye(4), [])
        r = np.linalg.norm(apply(D, picks[:, :3]) - picks[:, 3:], axis=1) if len(picks) else []
        rep = dict(mode=mode, chosen='pairs only', pairs=[dict(i=i + 1, residual=float(x), used_as_start=bool(u), suspect=not u)
                                                          for i, (x, u) in enumerate(zip(r, use))])
    after = stats(D, Xm, Xf)
    log(f'before: {before}\nafter:  {after}\ncorrection: {np.linalg.norm(D[:3, 3]):.3f} m, {rot_deg(D):.3f} deg')
    return dict(D=D.tolist(), before=before, after=after, shift=float(np.linalg.norm(D[:3, 3])), rotation_deg=rot_deg(D),
                scale=float(np.cbrt(abs(np.linalg.det(D[:3, :3])))), moving=Lm['id'], fixed=Lf['id'], report=rep)


def apply_to_layer(project, lid, D):
    P = project if isinstance(project, Project) else Project(project)
    L = P.layer(lid); L.setdefault('history', []).append(L['transform'])
    P.set_transform(lid, np.asarray(D, float) @ P.T(L))
    return dict(layer=P.layer(lid), note='press Prepare to rebuild the streamed data')
