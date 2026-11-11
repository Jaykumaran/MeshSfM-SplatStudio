"""A MeshSfM SplatStudio project: one folder holding everything the studio makes, plus project.json.

  <project>/
    project.json        name, frame, layers (meshes / splats / point clouds), path, settings
    mesh/               streaming chunks of the imported meshes (+ ground grid)
    tiles/              top-down orthophoto tiles (+ ground-surface points)
    edits/              non-destructive mesh edits (masks, vertex edits, patches, undo history)
    annotations/        lane lines and exports
    splat/              streaming tiles of splat layers
    exports/            exported meshes
    gen/                mesh-generation work files

Internal frame: right-handed, z up, metres. Every layer has a 4x4 'transform' that takes its own file
coordinates into this frame (e.g. a COLMAP / 3DGS splat that is y-down gets the y-down -> z-up rotation).
Source files are never modified; they are referenced by path.
"""
import json, os, time
import numpy as np

VERSION = 1

# common axis conventions -> internal z-up
FRAMES = {
    'z_up': np.eye(3),
    'y_up': np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]),     # (x, y, z) y-up (OpenGL / glTF)  -> z-up
    'y_down': np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]]),   # (x, y, z) y-down (COLMAP / 3DGS) -> z-up
}

DEFAULTS = dict(
    chunk=50.0,            # mesh chunk size (m) for streaming / editing
    ground_cell=0.5,       # ground height grid (m)
    ortho_px=0.025,        # orthophoto pixel (m)
    corridor_half=4.0,     # road half-width used when no lane edges are annotated (m)
)


def frame_matrix(name):
    M = np.eye(4); M[:3, :3] = FRAMES[name]; return M


class Project:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.file = os.path.join(self.root, 'project.json')
        if os.path.exists(self.file):
            self.data = json.load(open(self.file))
        else:
            self.data = dict(version=VERSION, name=os.path.basename(self.root), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                             layers=[], settings=dict(DEFAULTS), path=None, georef=None)
        for k, v in DEFAULTS.items():
            self.data.setdefault('settings', {}).setdefault(k, v)

    # ------------------------------------------------------------------ folders
    def dir(self, name, *more):
        d = os.path.join(self.root, name, *more)
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def mesh_dir(self): return self.dir('mesh')
    @property
    def tiles_dir(self): return self.dir('tiles')
    @property
    def edits_dir(self): return self.dir('edits')
    @property
    def ann_dir(self): return self.dir('annotations')
    @property
    def splat_dir(self): return self.dir('splat')
    @property
    def export_dir(self): return self.dir('exports')

    @property
    def settings(self): return self.data['settings']

    def save(self):
        os.makedirs(self.root, exist_ok=True)
        tmp = self.file + '.tmp'
        json.dump(self.data, open(tmp, 'w'), indent=1); os.replace(tmp, self.file)

    # ------------------------------------------------------------------ layers
    @property
    def layers(self): return self.data['layers']

    def layer(self, lid):
        for L in self.layers:
            if L['id'] == lid:
                return L
        raise KeyError(lid)

    def add_layer(self, kind, src, name=None, frame='z_up', transform=None, visible=True, **extra):
        """kind: 'mesh' (single file or folder of mesh files), 'splat' (3DGS .ply), 'points' (.ply / .las / COLMAP model)."""
        assert kind in ('mesh', 'splat', 'points'), kind
        T = np.asarray(transform, float) if transform is not None else frame_matrix(frame)
        n = sum(1 for L in self.layers if L['kind'] == kind) + 1
        lid = f'{kind}{n}'
        while any(L['id'] == lid for L in self.layers):
            n += 1; lid = f'{kind}{n}'
        L = dict(id=lid, kind=kind, name=name or os.path.basename(os.path.normpath(src)), src=os.path.abspath(src),
                 frame=frame, transform=T.tolist(), visible=bool(visible), **extra)
        self.layers.append(L); self.save()
        return L

    def files_of(self, L):
        """Mesh / splat files of a layer (a folder layer expands to its files, sorted)."""
        exts = {'mesh': ('.ply', '.obj', '.glb', '.gltf', '.stl', '.off'), 'splat': ('.ply',), 'points': ('.ply', '.las', '.laz', '.pcd', '.xyz', '.npz')}[L['kind']]
        if os.path.isdir(L['src']):
            return sorted(os.path.join(L['src'], f) for f in os.listdir(L['src']) if f.lower().endswith(exts))
        return [L['src']]

    @staticmethod
    def T(L):
        return np.asarray(L['transform'], float)

    def set_transform(self, lid, T):
        self.layer(lid)['transform'] = np.asarray(T, float).tolist(); self.save()

    def source_frame(self):
        """Transform internal z-up -> the frame of the first mesh layer (exports are written in both frames)."""
        for L in self.layers:
            if L['kind'] == 'mesh':
                return np.linalg.inv(self.T(L)), L['frame'] if np.allclose(self.T(L), frame_matrix(L['frame'])) else 'source'
        return np.eye(4), 'z_up'


def apply(T, X):
    """Apply a 4x4 transform to (n, 3) points."""
    T = np.asarray(T, float)
    return X @ T[:3, :3].T + T[:3, 3]


def apply_dir(T, N):
    """Rotate directions / normals by a 4x4 similarity transform (scale removed)."""
    R = np.asarray(T, float)[:3, :3]
    s = np.cbrt(abs(np.linalg.det(R))) or 1.0
    return N @ (R / s).T
