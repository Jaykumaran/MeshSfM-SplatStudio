"""HTTP server of MeshSfM SplatStudio (127.0.0.1 by default): the web UI + a small JSON / binary API."""
import glob, json, os, shutil, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import core
from .project import Project

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')


def short_paths(text):
    """Absolute paths in job messages shown in the browser -> '…/last/two' (screenshots and videos leak no local folders)."""
    import re
    return re.sub(r'(?<![\w.:/])/(?:[^\s/]+/)+[^\s/]*', lambda m: '…/' + '/'.join([x for x in m.group(0).split('/') if x][-2:]), text)


class App:
    """Holds the open project and its (re)loaded state; long tasks run one at a time as jobs."""
    def __init__(self, root):
        self.status = dict(busy=False, msg='', progress=0.0, result=None, error=None, log=[])
        self.lock = threading.RLock()
        self.open(root)

    def open(self, root):
        self.project = Project(root); self.project.save(); core.configure(self.project)
        self.reload()

    def reload(self):
        P = self.project
        self.route = core.Route() if os.path.exists(os.path.join(P.mesh_dir, 'path.npz')) else None
        self.store = None
        if self.route is not None and os.path.exists(os.path.join(P.mesh_dir, 'meta.json')):
            self.store = core.MeshStore(self.route); self.store.status = self.status
        self.surface = core.Surface(self.route) if self.route is not None and os.path.exists(os.path.join(P.tiles_dir, 'surface_xyz.npy')) else None
        tm = os.path.join(P.tiles_dir, 'meta.json')
        self.tmeta = json.load(open(tm)) if os.path.exists(tm) else None
        geo = os.path.join(P.root, 'georef.json')
        self.geo = json.load(open(geo)) if os.path.exists(geo) else None
        if self.geo and 'A' not in self.geo and 'lat0' in self.geo:      # plain local ENU origin -> the UI's georef form
            import math
            a_, e2 = 6378137.0, 6.69437999014e-3; s_ = math.sin(math.radians(self.geo['lat0']))
            self.geo.update(A=[[1, 0], [0, 1]], t=[0, 0], meridian_radius=a_ * (1 - e2) / (1 - e2 * s_ * s_) ** 1.5,
                            normal_radius=a_ / math.sqrt(1 - e2 * s_ * s_))

    def job(self, fn, label, reload=False):
        if self.status.get('busy'):
            return dict(ok=False, error='busy: ' + self.status.get('msg', ''))

        def work():
            self.status.update(busy=True, msg=label, progress=0.0, result=None, error=None, log=[])
            try:
                res = fn()
                if reload:
                    self.reload()
                self.status.update(result=res)
            except Exception as e:
                traceback.print_exc(); self.status.update(error=short_paths(repr(e)))
            self.status.update(busy=False, msg='', progress=1.0)
        threading.Thread(target=work, daemon=True).start()
        return dict(ok=True, started=label)

    def job_proc(self, kind, body, label, reload=False):
        """Heavy jobs (generate / prepare / align) in their own process: the server stays responsive."""
        import subprocess, sys, tempfile
        if self.status.get('busy'):
            return dict(ok=False, error='busy: ' + self.status.get('msg', ''))
        if kind in ('prepare', 'gen', 'path_from_line'):
            self.store = None; import gc; gc.collect()          # release memory-mapped chunks
        tmp = tempfile.mkdtemp(prefix='meshsfm_job_')
        jf, out = os.path.join(tmp, 'job.json'), os.path.join(tmp, 'result.json')
        json.dump(dict(kind=kind, project=self.project.root, body=body, out=out), open(jf, 'w'))

        def work():
            self.status.update(busy=True, msg=label, progress=0.0, result=None, error=None, log=[])
            try:
                p = subprocess.Popen([sys.executable, '-m', 'meshsfm', 'job', jf], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                for line in p.stdout:
                    line = line.rstrip()
                    if line and 'Warning' not in line:
                        line = short_paths(line)
                        self.status['log'] = (self.status['log'] + [line])[-200:]; self.status['msg'] = f'{label}: {line[:120]}'
                rc = p.wait()
                if rc != 0:
                    raise RuntimeError('job failed: ' + ' | '.join(self.status['log'][-3:]))
                self.project = Project(self.project.root); core.configure(self.project)
                if reload:
                    self.reload()
                self.status.update(result=json.load(open(out)).get('result'))
            except Exception as e:
                traceback.print_exc(); self.status.update(error=short_paths(repr(e)))
                if reload:
                    try:
                        self.reload()
                    except Exception:
                        pass
            self.status.update(busy=False, msg='', progress=1.0)
        threading.Thread(target=work, daemon=True).start()
        return dict(ok=True, started=label)

    # ---------------------------------------------------------------- views
    def meta(self):
        P = self.project
        route = []
        if self.route is not None:
            r = self.route; step = max(1, int(round(2.0 / max(1e-3, np.median(np.diff(r.arc))))))
            route = np.c_[r.P[::step, 0], r.P[::step, 1], r.arc[::step], r.P[::step, 2]].round(3).tolist()
        base = dict(self.tmeta) if self.tmeta else dict(tiles={}, px0=core.PX0, tile=core.T, levels=0, x0=0, ztop=0)
        if not route:      # empty project: a 1 m placeholder so the map has an origin
            route = [[0, 0, 0, 0], [1, 0, 1, 0]]
        return dict(base, route=route, auto_available=False, geo=self.geo, mesh=bool(self.store), project=self.project_info())

    def project_info(self):
        P = self.project
        spl = {}
        for L in P.layers:
            f = os.path.join(P.root, 'splat', L['id'], 'meta.json')
            if L['kind'] == 'splat' and os.path.exists(f):
                spl[L['id']] = json.load(open(f))
        return dict(root=P.root, name=P.data.get('name'), layers=P.layers, settings=P.settings, path=P.data.get('path'),
                    prepared=P.data.get('prepared'), splats=spl, generated=[k for k, v in P.data.get('generated', {}).items() if v.get('capture')], origin=(self.store.O.tolist() if self.store else
                                                                         (self.route.origin.tolist() if self.route is not None else [0, 0, 0])))


def list_dir(path):
    path = os.path.abspath(os.path.expanduser(path or '~'))
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    out = []
    try:
        for n in sorted(os.listdir(path), key=str.lower):
            if n.startswith('.'):
                continue
            f = os.path.join(path, n)
            out.append(dict(name=n, dir=os.path.isdir(f), size=os.path.getsize(f) if os.path.isfile(f) else 0))
    except PermissionError:
        pass
    return dict(path=path, parent=os.path.dirname(path), entries=out)


def serve(root, host='127.0.0.1', port=8780):
    app = App(root)

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def _send(self, body, ctype='application/json', code=200, cache=False):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=86400' if cache else 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                self._get()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                traceback.print_exc(); self._send(dict(error=repr(e)), code=500)

        def do_POST(self):
            try:
                n = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(n) or b'{}')
                self._post(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                traceback.print_exc(); self._send(dict(ok=False, error=repr(e)), code=500)

        # ------------------------------------------------------------ GET
        def _get(self):
            from urllib.parse import unquote
            pth, _, q = self.path.partition('?')
            qs = {k: unquote(v) for k, v in (kv.split('=', 1) for kv in q.split('&') if '=' in kv)}
            store = app.store
            if pth in ('/', '/index.html'):
                return self._send(open(os.path.join(WEB, 'index.html'), 'rb').read(), 'text/html')
            if pth.startswith('/web/'):
                f = os.path.join(WEB, os.path.basename(pth))
                ct = 'text/javascript' if f.endswith('.js') else 'text/css' if f.endswith('.css') else 'image/svg+xml' if f.endswith('.svg') else 'text/plain'
                return self._send(open(f, 'rb').read(), ct)
            if pth == '/meta':
                return self._send(app.meta())
            if pth == '/project':
                return self._send(app.project_info())
            if pth == '/fs/ls':
                return self._send(list_dir(qs.get('path')))
            if pth == '/job':
                return self._send({k: v for k, v in app.status.items()})
            if pth == '/annotation':
                f = os.path.join(core.ANN, 'annotation.json')
                return self._send(json.load(open(f)) if os.path.exists(f) else {'lines': [], 'reviewed': []})
            if pth.startswith('/tile/'):
                _, _, L, name = pth.split('/')
                f = os.path.join(core.TILES, L, os.path.basename(name))
                return self._send(open(f, 'rb').read(), 'image/jpeg', cache=False) if os.path.exists(f) else self._send(b'', 'text/plain', 404)
            if pth.startswith('/splat/tile/'):
                _, _, _, lid, name = pth.split('/')
                f = os.path.join(app.project.root, 'splat', os.path.basename(lid), os.path.basename(name) + '.bin')
                return self._send(open(f, 'rb').read(), 'application/octet-stream') if os.path.exists(f) else self._send(b'', 'text/plain', 404)
            if pth == '/route3d':
                return self._send(self._route3d())
            if not store and pth == '/mesh/status':
                st = {k: v for k, v in app.status.items() if k != 'log'}
                return self._send(dict(st, version=0, changed=[], geom=[], proposal=None, summary=dict(deleted=0, moved=0, patches=0, patch_tris=0, total=0, undo=[], redo=[])))
            if not store:
                return self._send(dict(error='no mesh yet: add a mesh layer (or generate one) and press Prepare'), code=404)
            if pth == '/mesh/meta':
                return self._send(dict(origin=store.O.tolist(), chunks=list(store.chunks.values()), version=store.version,
                                       parts=store.meta['parts']))
            if pth.startswith('/mesh/chunk/'):
                return self._send(store.chunk_bytes(pth.split('/')[-1], int(qs.get('lod', 0))), 'application/octet-stream')
            if pth.startswith('/mesh/vdelta/'):
                return self._send(store.vdelta_bytes(pth.split('/')[-1]), 'application/octet-stream')
            if pth.startswith('/mesh/mask/'):
                return self._send(store.mask_bytes(pth.split('/')[-1], qs.get('which', 'del')), 'application/octet-stream')
            if pth == '/mesh/patches':
                return self._send(store.patch_list())
            if pth.startswith('/mesh/patch/'):
                return self._send(store.patch_bytes(pth.split('/')[-1]), 'application/octet-stream')
            if pth == '/mesh/status':
                try:
                    since = int(qs.get('since', 0))
                except ValueError:
                    since = 0
                changed = sorted({k for v, ks in list(store.changes) if v > since for k in ks})
                prop = store.proposal and dict(label=store.proposal['label'], stats=store.proposal['stats'], tool=store.proposal['tool'])
                st = {k: v for k, v in app.status.items() if k != 'log'}
                geom = [k for k in changed if k in store.touched]
                return self._send(dict(st, version=store.version, changed=changed, geom=geom, proposal=prop, summary=store.summary()))
            if pth == '/ground':
                xy = np.array([[float(qs['x']), float(qs['y'])]])
                return self._send(dict(z=float(store.ground_at(xy)[0])))
            return self._send(b'not found', 'text/plain', 404)

        def _route3d(self):
            store, route = app.store, app.route
            if route is None:
                return dict(route=[], lines=[])
            O = store.O if store else route.origin
            step = max(1, int(round(2.0 / max(1e-3, np.median(np.diff(route.arc))))))
            out = dict(route=(route.P[::step] - O).round(3).tolist(), lines=[])
            f = os.path.join(core.ANN, 'annotation.json')
            if os.path.exists(f) and app.surface is not None:
                for ln in json.load(open(f)).get('lines', []):
                    if len(ln['nodes']) < 2 or not ln.get('visible', True):
                        continue
                    xy = core.resample(core.catmull_rom(ln['nodes'], ln.get('smooth', True)), 0.5)
                    X = np.c_[xy, app.surface.heights(xy, smooth=2)] - O
                    out['lines'].append(dict(type=ln['type'], pts=X.round(3).tolist()))
            return out

        # ------------------------------------------------------------ POST
        def _post(self, body):
            P = app.project
            p = self.path
            # ---- project / layers
            if p == '/project/open':
                app.open(body['root']); return self._send(dict(ok=True, project=app.project_info()))
            if p == '/project/layer/add':
                L = P.add_layer(body['kind'], body['src'], name=body.get('name'), frame=body.get('frame', 'z_up'))
                return self._send(dict(ok=True, layer=L))
            if p == '/project/layer/update':
                L = P.layer(body['id'])
                for k in ('name', 'visible'):
                    if k in body:
                        L[k] = body[k]
                if 'frame' in body:
                    from .project import frame_matrix
                    L['frame'] = body['frame']; L['transform'] = frame_matrix(body['frame']).tolist()
                if 'transform' in body:
                    L['transform'] = body['transform']
                P.save(); return self._send(dict(ok=True, layer=L))
            if p == '/project/layer/remove':
                P.data['layers'] = [L for L in P.layers if L['id'] != body['id']]; P.save()
                return self._send(dict(ok=True))
            if p == '/project/settings':
                P.settings.update({k: float(v) for k, v in body.items()}); P.save(); core.configure(P)
                return self._send(dict(ok=True, settings=P.settings))
            if p == '/project/prepare':
                from . import prep
                if body.get('reset_path'):
                    f = os.path.join(P.mesh_dir, 'path.npz')
                    if os.path.exists(f):
                        os.remove(f)
                return self._send(app.job_proc('prepare', {}, 'preparing layers', reload=True))
            # ---- generate
            if p == '/gen/run':
                return self._send(app.job_proc('gen', body, f"generating mesh ({body.get('pipeline')})", reload=True))
            if p == '/splat/export_colmap':
                return self._send(app.job_proc('export_colmap', body, 'exporting a COLMAP dataset for splat training'))
            # ---- align
            if p == '/align/run':
                return self._send(app.job_proc('align', body, 'aligning layers'))
            if p == '/align/apply':
                from . import align
                res = align.apply_to_layer(P, body['id'], body['T'])
                return self._send(dict(ok=True, **res))
            # ---- annotation
            if p == '/heights':
                out = []
                for ln in body.get('lines', []):
                    pts = np.asarray(ln.get('pts', []), float).reshape(-1, 2); nodes = np.asarray(ln.get('nodes', []), float).reshape(-1, 2)
                    hs = app.surface
                    f = (lambda X, sm: hs.heights(X, smooth=sm).round(3).tolist()) if hs else (lambda X, sm: [0.0] * len(X))
                    out.append(dict(pts=f(pts, 2) if len(pts) else [], nodes=f(nodes, 0) if len(nodes) else []))
                return self._send(dict(heights=out))
            if p == '/save':
                ann_file = os.path.join(core.ANN, 'annotation.json')
                body['saved'] = time.strftime('%Y-%m-%d %H:%M:%S'); body['frame'] = 'z_up'
                os.makedirs(os.path.join(core.ANN, 'backups'), exist_ok=True)
                if os.path.exists(ann_file):
                    shutil.copy(ann_file, os.path.join(core.ANN, 'backups', time.strftime('annotation_%Y%m%d_%H%M%S.json')))
                    for old in sorted(glob.glob(os.path.join(core.ANN, 'backups', '*.json')))[:-30]:
                        os.remove(old)
                json.dump(body, open(ann_file + '.tmp', 'w')); os.replace(ann_file + '.tmp', ann_file)
                info = {}
                if body.get('export', True) and app.surface is not None:
                    try:
                        info = core.export_lanes(app.surface, body)
                    except Exception as e:
                        info = {'export_error': repr(e)}
                return self._send(dict(ok=True, saved=body['saved'], **info))
            if p == '/path/from_lines':      # use an annotated line (e.g. a centre line) as the drive path
                from . import prep
                ann = json.load(open(os.path.join(core.ANN, 'annotation.json')))
                ln = [l for l in ann['lines'] if l['id'] == body['id']][0]
                xy = core.resample(core.catmull_rom(ln['nodes'], ln.get('smooth', True)), 0.5)
                z = app.surface.heights(xy) if app.surface else np.zeros(len(xy))
                Pp, arc, F = prep.path_from_points(np.c_[xy, z]); prep.save_path(P, Pp, arc, F, f"line {ln['id']}")
                return self._send(app.job_proc('path_from_line', {}, 're-chunking along the new path', reload=True))
            # ---- mesh edits
            store = app.store
            if not store:
                return self._send(dict(ok=False, error='no mesh yet'), code=404)
            with store.lock:
                if p == '/mesh/edit':
                    return self._send(store.commit(body.get('label', 'manual edit'), dele=body.get('del'), undel=body.get('undel')))
                if p == '/mesh/undo':
                    return self._send(store.undo())
                if p == '/mesh/redo':
                    return self._send(store.redo_())
                if p == '/mesh/accept':
                    return self._send(store.accept())
                if p == '/mesh/discard':
                    return self._send(store.discard())
                if p == '/mesh/propose_ids':
                    dele = {k: np.asarray(v, np.int64) for k, v in body.get('del', {}).items() if len(v)}
                    n = int(sum(len(v) for v in dele.values()))
                    store.proposal = {'tool': 'select', 'label': body.get('label', 'selection'), 'del': dele, 'patches': [],
                                      'stats': dict(delete=n, patch_tris=0, chunks=len(dele))}
                    store._touch(list(dele))
                    return self._send(dict(ok=True, **store.proposal['stats']))
                if p == '/mesh/blob':
                    sel, msg = store.blob(body['key'], int(body['tri']), float(body.get('hprotect', 0.3)))
                    if sel is None:
                        return self._send(dict(ok=False, error=msg))
                    store.proposal = {'tool': 'blob', 'label': 'picked blob', 'del': sel, 'patches': [],
                                      'stats': dict(delete=int(sum(len(v) for v in sel.values())), patch_tris=0, chunks=len(sel))}
                    store._touch(list(sel))
                    return self._send(dict(ok=True, msg=msg, **store.proposal['stats']))
                if p == '/mesh/smooth':
                    return self._send(store.smooth(body['pts'], body.get('params', {}), body.get('stroke'), bool(body.get('final')),
                                                   bool(body.get('close_only'))))
                if p == '/mesh/crop':          # polygon (z-up xy): delete inside or outside, as a proposal
                    return self._send(store.crop(body['poly'], body.get('mode', 'outside'), body.get('zmin'), body.get('zmax')))
            if p == '/mesh/propose':
                return self._send(app.job(lambda: store.propose(body['tool'], body.get('params', {})), f"computing {body['tool']}"))
            if p == '/mesh/fill':
                return self._send(app.job(lambda: store.fill(body['poly'], body.get('params', {})), 'computing fill'))
            if p == '/mesh/export':
                return self._send(app.job(lambda: store.export(bool(body.get('obj'))), 'exporting mesh'))
            if p == '/mesh/segment_road':
                return self._send(app.job(lambda: store.export_road_segmented(body), 'segmenting road & saving PLY'))
            if p == '/mesh/recolour':
                return self._send(app.job(lambda: store.recolour_ground(poly=body.get('poly')), 're-projecting colours'))
            return self._send(b'not found', 'text/plain', 404)

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    print(f'MeshSfM SplatStudio: http://{host}:{port}   project {app.project.root}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
