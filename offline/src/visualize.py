"""Export and inspect point clouds and grasp axes in Open3D."""
from pathlib import Path

import numpy as np


def geometries(points, colors, target_mask, grasps):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    shown = np.array(colors, copy=True)
    shown[target_mask] = 0.45 * shown[target_mask] + 0.55 * np.array([1.0, 0.75, 0.0])
    cloud.colors = o3d.utility.Vector3dVector(shown.astype(float))
    out = [cloud]
    for grasp in grasps:
        p = np.asarray(grasp["position_xyz_m"], float)
        r = np.asarray(grasp["rotation_matrix"], float)
        score = float(grasp["model_score"])
        width = float(grasp["opening_width_m"])
        # Red: approach; blue: closing; green: third grasp-frame axis.
        line_points = np.vstack((p, p + .06*r[:, 0], p - .5*width*r[:, 1],
                                 p + .5*width*r[:, 1], p + .03*r[:, 2]))
        lines = [[0, 1], [2, 3], [0, 4]]
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(line_points)
        axis.lines = o3d.utility.Vector2iVector(lines)
        intensity = np.clip(score, 0.2, 1.0)
        axis.colors = o3d.utility.Vector3dVector([[intensity, 0, 0], [0, 0, intensity], [0, intensity, 0]])
        out.append(axis)
    return out


def show(points, colors, target_mask, grasps):
    import open3d as o3d
    o3d.visualization.draw_geometries(geometries(points, colors, target_mask, grasps),
                                      window_name="AnyGrasp: gold target, red approach, blue closing")


def save_lines(path, points, colors, target_mask, grasps):
    import open3d as o3d
    geoms = geometries(points, colors, target_mask, grasps)
    combined = o3d.geometry.LineSet()
    for item in geoms[1:]:
        combined += item
    if grasps:
        o3d.io.write_line_set(str(Path(path)), combined)


def save_html(path, points, colors, target_mask, grasps):
    """Self-contained interactive 3D scene; scores appear on candidate hover."""
    import plotly.graph_objects as go

    scene_ids = np.flatnonzero(~target_mask)[::max(1, int((~target_mask).sum() / 18000))]
    target_ids = np.flatnonzero(target_mask)[::max(1, int(target_mask.sum() / 12000))]
    traces = []
    for ids, name, tint, size in ((scene_ids, "Scene", False, 1.3),
                                  (target_ids, "Selected target", True, 2.5)):
        p = points[ids]
        if not len(p):
            continue
        rgb = np.clip(colors[ids] * 255, 0, 255).astype(np.uint8)
        marker_colors = "gold" if tint else [f"rgb({r},{g},{b})" for r, g, b in rgb]
        traces.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   name=name, marker=dict(size=size, color=marker_colors, opacity=0.8)))
    for i, grasp in enumerate(grasps):
        p = np.asarray(grasp["position_xyz_m"])
        r = np.asarray(grasp["rotation_matrix"])
        width = float(grasp["opening_width_m"])
        for label, a, b, color in (
            ("approach", p, p + 0.06*r[:, 0], "red"),
            ("closing", p - width/2*r[:, 1], p + width/2*r[:, 1], "blue")):
            traces.append(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                                       mode="lines+markers", name=f"#{i+1} {label}",
                                       line=dict(color=color, width=7), marker=dict(size=3, color=color),
                                       hovertemplate=f"Candidate {i+1}<br>score {grasp['model_score']:.3f}<br>width {width:.3f} m<extra></extra>"))
    fig = go.Figure(traces)
    fig.update_layout(title="AnyGrasp offline scene (camera optical frame, metres)",
                      scene=dict(xaxis_title="X right (m)", yaxis_title="Y down (m)",
                                 zaxis_title="Z forward (m)", aspectmode="data"),
                      legend=dict(itemsizing="constant"))
    fig.write_html(str(Path(path)), include_plotlyjs=True)
