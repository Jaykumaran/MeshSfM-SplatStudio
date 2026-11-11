#!/usr/bin/env bash
# Downloads one KITTI raw drive (2011_09_30_drive_0027, ~4.4 GB) and builds the two demo projects:
#   <out>/projects/kitti_lidar_rgb_gps   LiDAR + RGB + GPS/INS
#   <out>/projects/kitti_rgb_gps         RGB + GPS/INS (COLMAP)
# KITTI raw is CC BY-NC-SA 3.0 (non-commercial): http://www.cvlibs.net/datasets/kitti/raw_data.php
# usage: scripts/get_kitti_demo.sh <out_dir> [first_frame:last_frame]
set -e
OUT=$(readlink -f "${1:?usage: get_kitti_demo.sh OUT_DIR [0:200]}"); FR=${2:-0:200}
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p "$OUT/kitti_raw" && cd "$OUT/kitti_raw"
U=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
[ -f 2011_09_30_calib.zip ] || curl -fO $U/2011_09_30_calib.zip
[ -f 2011_09_30_drive_0027_sync.zip ] || curl -fO $U/2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip
unzip -qn 2011_09_30_calib.zip; unzip -qn 2011_09_30_drive_0027_sync.zip
cd - >/dev/null
CAP="$OUT/captures/kitti_0027_${FR/:/_}"
./meshsfm.sh kitti "$OUT/kitti_raw/2011_09_30" 2011_09_30_drive_0027 "$CAP" --frames "$FR" --masks
./meshsfm.sh gen "$OUT/projects/kitti_lidar_rgb_gps" lidar "$CAP" --name lidar_rgb_gps
./meshsfm.sh gen "$OUT/projects/kitti_rgb_gps" rgb "$CAP" --name rgb_gps
echo "open:  ./meshsfm.sh $OUT/projects/kitti_lidar_rgb_gps    (or kitti_rgb_gps)"
