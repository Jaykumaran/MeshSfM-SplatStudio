"""KITTI raw -> capture folder (see gen.py): left colour camera (image_02, rectified), Velodyne scans, OXTS GPS/IMU poses.

  python -m meshsfm.datasets.kitti <kitti_raw>/2011_09_30 2011_09_30_drive_0027 <capture_out> --frames 0:200 [--masks]

The local frame is east-north-up (metres) with the origin at the first frame; georef.json keeps its lat / lon / alt.
KITTI raw is CC BY-NC-SA 3.0 (http://www.cvlibs.net/datasets/kitti/); download it yourself, it is not redistributed.
"""
import argparse, json, os, shutil
import numpy as np


def read_calib(path):
    out = {}
    for line in open(path):
        if ':' in line:
            k, v = line.split(':', 1)
            try:
                out[k] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    return out


def rt(c):
    T = np.eye(4); T[:3, :3] = c['R'].reshape(3, 3); T[:3, 3] = c['T']; return T


def oxts_poses(rows):
    """KITTI devkit convention: Mercator x / y from lat / lon, z = altitude, R = Rz(yaw) Ry(pitch) Rx(roll) (IMU -> world)."""
    er = 6378137.0; lat0 = rows[0][0]; s = np.cos(np.radians(lat0))
    out = []
    for r in rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        tx = s * er * np.radians(lon); ty = s * er * np.log(np.tan(np.pi * (90 + lat) / 360)); tz = alt
        cx, sx, cy, sy, cz, sz = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = [tx, ty, tz]; out.append(T)
    O = out[0][:3, 3].copy()
    for T in out:
        T[:3, 3] -= O
    return out


def auto_masks(images, out_dir):
    """Cars, people, bikes, buses masked with a small pretrained segmentation net (torchvision DeepLabV3, VOC classes)."""
    import cv2, torch, torchvision
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    try:
        net = deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    except TypeError:
        net = deeplabv3_mobilenet_v3_large(pretrained=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; net = net.eval().to(dev)
    dyn = [2, 6, 7, 14, 15, 19]   # VOC: bicycle, bus, car, motorbike, person, train
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    os.makedirs(out_dir, exist_ok=True); n = 0
    with torch.no_grad():
        for f in images:
            img = cv2.imread(f)[:, :, ::-1].copy()
            x = (torch.from_numpy(img).to(dev).permute(2, 0, 1).float() / 255 - mean) / std
            lab = net(x[None])['out'][0].argmax(0).cpu().numpy()
            m = np.isin(lab, dyn).astype(np.uint8) * 255
            m = cv2.dilate(m, np.ones((9, 9), np.uint8))
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '.png'), m); n += (m > 0).mean()
    print(f'auto masks: {len(images)} images, {n / max(1, len(images)):.1%} of pixels masked on average', flush=True)


def convert(day_dir, drive, out, frames=(0, 200), masks=False, step=1, stereo=False):
    d = os.path.join(day_dir, drive + '_sync')
    cc = read_calib(os.path.join(day_dir, 'calib_cam_to_cam.txt'))
    T_cam0_velo = rt(read_calib(os.path.join(day_dir, 'calib_velo_to_cam.txt')))
    T_velo_imu = rt(read_calib(os.path.join(day_dir, 'calib_imu_to_velo.txt')))
    P2 = cc['P_rect_02'].reshape(3, 4); K = P2[:, :3]; t2 = np.linalg.solve(K, P2[:, 3])
    Rr = np.eye(4); Rr[:3, :3] = cc['R_rect_00'].reshape(3, 3)
    T_c2_c0 = np.eye(4); T_c2_c0[:3, 3] = t2
    T_c2_velo = T_c2_c0 @ Rr @ T_cam0_velo
    P3 = cc['P_rect_03'].reshape(3, 4); T_c3_c0 = np.eye(4); T_c3_c0[:3, 3] = np.linalg.solve(P3[:, :3], P3[:, 3])
    T_c3_velo = T_c3_c0 @ Rr @ T_cam0_velo
    names = sorted(os.listdir(os.path.join(d, 'oxts', 'data')))[frames[0]:frames[1]:step]
    rows = [np.loadtxt(os.path.join(d, 'oxts', 'data', n)) for n in names]
    T_w_imu = oxts_poses(rows)
    for sub in ('images', 'lidar'):
        p = os.path.join(out, sub); shutil.rmtree(p, ignore_errors=True); os.makedirs(p)
    import cv2
    h, w = cv2.imread(os.path.join(d, 'image_02', 'data', names[0].replace('.txt', '.png'))).shape[:2]
    fr = []
    for n, T in zip(names, T_w_imu):
        b = n.replace('.txt', '')
        shutil.copy(os.path.join(d, 'image_02', 'data', b + '.png'), os.path.join(out, 'images', b + '.png'))
        shutil.copy(os.path.join(d, 'velodyne_points', 'data', b + '.bin'), os.path.join(out, 'lidar', b + '.bin'))
        T_w_velo = T @ np.linalg.inv(T_velo_imu)
        T_w_cam = T_w_velo @ np.linalg.inv(T_c2_velo)
        r = rows[names.index(n)]
        fr.append(dict(image=b + '.png', lidar=b + '.bin', T_world_cam=T_w_cam.round(6).tolist(), T_world_lidar=T_w_velo.round(6).tolist(),
                       gps=[float(r[0]), float(r[1]), float(r[2])]))
        if stereo:             # right colour camera (image_03): same rectified intrinsics, its own baseline
            shutil.copy(os.path.join(d, 'image_03', 'data', b + '.png'), os.path.join(out, 'images', 'r_' + b + '.png'))
            T_w_cam3 = T_w_velo @ np.linalg.inv(T_c3_velo)
            fr.append(dict(image='r_' + b + '.png', lidar=b + '.bin', T_world_cam=T_w_cam3.round(6).tolist(), T_world_lidar=T_w_velo.round(6).tolist(),
                           gps=[float(r[0]), float(r[1]), float(r[2])], camera='right'))
    json.dump(dict(K=K.tolist(), width=int(w), height=int(h), frames=fr, source=f'KITTI raw {drive} frames {frames[0]}-{frames[1]}'),
              open(os.path.join(out, 'cameras.json'), 'w'), indent=1)
    json.dump(dict(lat0=float(rows[0][0]), lon0=float(rows[0][1]), alt0=float(rows[0][2]), frame='local ENU-like (KITTI Mercator), z up'),
              open(os.path.join(out, 'georef.json'), 'w'))
    if masks:
        auto_masks(sorted(os.path.join(out, 'images', f) for f in os.listdir(os.path.join(out, 'images'))), os.path.join(out, 'masks'))
    print(f'capture: {len(fr)} frames ({w}x{h}) -> {out}', flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('day_dir'); ap.add_argument('drive'); ap.add_argument('out')
    ap.add_argument('--frames', default='0:200'); ap.add_argument('--step', type=int, default=1); ap.add_argument('--masks', action='store_true'); ap.add_argument('--stereo', action='store_true', help='also the right colour camera')
    a = ap.parse_args(); f0, f1 = (int(x) for x in a.frames.split(':'))
    convert(a.day_dir, a.drive, a.out, (f0, f1), a.masks, a.step, a.stereo)
