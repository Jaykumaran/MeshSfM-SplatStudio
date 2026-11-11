"""MeshSfM SplatStudio command line.

  meshsfm serve   PROJECT [--port 8780]                 open the studio (creates an empty project if needed)
  meshsfm add     PROJECT mesh|splat|points PATH [--frame z_up|y_up|y_down] [--name N]
  meshsfm prepare PROJECT                                build chunks, ground, orthophoto, splat tiles
  meshsfm gen     PROJECT lidar|rgb|points CAPTURE [--no-masks] [--name N]
  meshsfm align   PROJECT MOVING_ID FIXED_ID [--apply]
  meshsfm kitti   KITTI_DAY_DIR DRIVE CAPTURE_OUT [--frames 0:200] [--masks]
"""
import argparse, json, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='meshsfm', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve'); s.add_argument('project'); s.add_argument('--port', type=int, default=8780); s.add_argument('--host', default='127.0.0.1')
    s = sub.add_parser('add'); s.add_argument('project'); s.add_argument('kind', choices=['mesh', 'splat', 'points']); s.add_argument('path')
    s.add_argument('--frame', default='z_up', choices=['z_up', 'y_up', 'y_down']); s.add_argument('--name')
    s = sub.add_parser('prepare'); s.add_argument('project')
    s = sub.add_parser('gen'); s.add_argument('project'); s.add_argument('pipeline', choices=['lidar', 'rgb', 'points'])
    s.add_argument('capture', nargs='?', help='capture folder (or give the inputs separately below)')
    for k in ('images', 'masks', 'lidar', 'poses', 'lidar-poses', 'timestamps', 'lidar-timestamps', 'colmap', 'points', 'gps', 'intrinsics', 'T-cam-lidar'):
        s.add_argument('--' + k)
    s.add_argument('--no-masks', action='store_true'); s.add_argument('--name'); s.add_argument('--no-objects', action='store_true')
    s.add_argument('--fast', action='store_true', help='RGB: photometric depth only (no geometric pass)'); s.add_argument('--ref-step', type=int, default=1, help='RGB: depth maps for every n-th image')
    s = sub.add_parser('align'); s.add_argument('project'); s.add_argument('moving'); s.add_argument('fixed'); s.add_argument('--apply', action='store_true')
    s.add_argument('--max-dist', type=float, default=1.0)
    s.add_argument('--mode', default='rigid', choices=['rigid', 'yaw', 'similarity', 'translation'], help='rigid | yaw (keep up direction) | similarity (+ scale) | translation')
    s = sub.add_parser('kitti'); s.add_argument('day_dir'); s.add_argument('drive'); s.add_argument('out'); s.add_argument('--frames', default='0:200')
    s.add_argument('--masks', action='store_true'); s.add_argument('--stereo', action='store_true', help='also the right colour camera')
    s = sub.add_parser('job', help='internal: run a studio job from a JSON file in its own process'); s.add_argument('file')
    a = ap.parse_args(argv)
    from .project import Project
    if a.cmd == 'serve':
        from .server import serve
        serve(a.project, a.host, a.port)
    elif a.cmd == 'add':
        P = Project(a.project); print(json.dumps(P.add_layer(a.kind, a.path, name=a.name, frame=a.frame), indent=1))
    elif a.cmd == 'prepare':
        from . import prep
        prep.prepare(Project(a.project))
    elif a.cmd == 'gen':
        from . import gen
        inputs = {k: getattr(a, k.replace('-', '_')) for k in ('images', 'masks', 'lidar', 'poses', 'lidar_poses', 'timestamps', 'lidar_timestamps', 'colmap', 'points', 'gps', 'intrinsics', 'T_cam_lidar')
                  if getattr(a, k.replace('-', '_'), None)}
        print(gen.run(Project(a.project), dict(pipeline=a.pipeline, capture=a.capture, inputs=inputs or None, masks=not a.no_masks,
                                              name=a.name or a.pipeline, objects=not a.no_objects, fast=a.fast, ref_step=a.ref_step)))
    elif a.cmd == 'align':
        from . import align
        P = Project(a.project); r = align.run(P, dict(moving=a.moving, fixed=a.fixed, max_dist=a.max_dist, mode=a.mode))
        if a.apply:
            align.apply_to_layer(P, a.moving, r['D']); print('applied; run "meshsfm prepare" to rebuild')
    elif a.cmd == 'job':
        J = json.load(open(a.file)); P = Project(J['project']); res = None
        if J['kind'] == 'prepare':
            from . import prep
            prep.prepare(P)
        elif J['kind'] == 'gen':
            from . import gen
            res = gen.run(P, J['body'])
        elif J['kind'] == 'align':
            from . import align
            res = align.run(P, J['body'])
        elif J['kind'] == 'export_colmap':
            import os
            from . import export_colmap
            B = J['body']; G = P.data['generated'][B['name']]
            out = os.path.join(P.root, 'gen', B['name'] + '_colmap_for_splat')
            res = dict(path=export_colmap.export(G['capture'], G['points'], out))
        elif J['kind'] == 'path_from_line':
            from . import prep
            prep.prepare(P)
        json.dump(dict(result=res), open(J['out'], 'w'))
    elif a.cmd == 'kitti':
        from .datasets import kitti
        f0, f1 = (int(x) for x in a.frames.split(':')); kitti.convert(a.day_dir, a.drive, a.out, (f0, f1), a.masks, stereo=a.stereo)


if __name__ == '__main__':
    main()
