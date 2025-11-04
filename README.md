# 3d-gaussian-splatting

3DGS Pipeline with multiple videos that runs efficiently on GPU

## Notes:
1) Video SFM Pose is ONLY **world-to-camera**
2) COLMAP's focal `length = 1.2 * max(width, height)` where width, height are in pixels works **VERY** well.
