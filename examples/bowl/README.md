# Object-only real capture

`target_points.npz`: `points_m`, N×3 float32, camera optical frame (X right, Y down, Z forward), metres. 5344 real target points retained; full RGB-D scene withheld for privacy.

`results.json`: historical model candidates and numerical experiment statistics, with source paths removed. `scene_points` and `model_points` describe the original private experiment, not the size of this public subset. No bowl grasp success is claimed.
