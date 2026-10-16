"""Interactive object region selection; no object-specific grasp geometry."""
import numpy as np


def mask_from_click(depth_mm, x, y, depth_tolerance_mm=35):
    """Connected same-depth region seeded by a user-selected pixel."""
    import cv2

    h, w = depth_mm.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Click outside image")
    seed_depth = float(depth_mm[y, x])
    if not np.isfinite(seed_depth) or seed_depth <= 0:
        raise ValueError("Clicked pixel has invalid depth")
    region = np.abs(np.asarray(depth_mm, np.float32) - seed_depth) <= depth_tolerance_mm
    _, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    selected = labels == labels[y, x]
    if selected.sum() < 30:
        raise RuntimeError("Selected region has fewer than 30 valid pixels")
    return selected


def mask_from_polygon(depth_mm, vertices):
    """Bound a manually selected object surface without depth flood filling."""
    import cv2
    vertices = np.asarray(vertices)
    h, w = depth_mm.shape
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("Select at least three contour points")
    if not np.isfinite(vertices).all() or (vertices < 0).any() or (vertices[:, 0] >= w).any() or (vertices[:, 1] >= h).any():
        raise ValueError("Polygon outside image")
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [vertices.astype(np.int32)], 1)
    valid = region.astype(bool) & np.isfinite(depth_mm) & (depth_mm > 0)
    if valid.sum() < 30:
        raise RuntimeError("Target region has fewer than 30 valid pixels")
    return valid


def choose_mask(rgb, depth_mm, mode="rectangle", depth_tolerance_mm=35):
    """Show the selected region and require explicit confirmation before inference."""
    import cv2
    while True:
        region = _choose_mask_once(rgb, depth_mm, mode, depth_tolerance_mm)
        preview = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        preview[region] = (0.5 * preview[region] + 0.5 * np.array([0, 210, 255])).astype(np.uint8)
        title = "Confirm yellow TARGET only: Enter accept, R reselect, Esc cancel"
        cv2.namedWindow(title)
        try:
            while True:
                cv2.imshow(title, preview)
                key = cv2.waitKey(30) & 0xff
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 or key == 27:
                    raise RuntimeError("Target selection cancelled")
                if key in (10, 13):
                    return region
                if key in (ord('r'), ord('R')):
                    break
        finally:
            cv2.destroyWindow(title)


def _choose_mask_once(rgb, depth_mm, mode, depth_tolerance_mm):
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = depth_mm.shape
    if mode == "polygon":
        title = "Outline TARGET: left click points, right click undo, Enter finish, Esc cancel"
        vertices = []
        def polygon_callback(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                vertices.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and vertices:
                vertices.pop()
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, polygon_callback)
        try:
            while True:
                preview = bgr.copy()
                for point in vertices:
                    cv2.circle(preview, point, 3, (0, 210, 255), -1)
                if len(vertices) >= 2:
                    cv2.polylines(preview, [np.asarray(vertices, np.int32)], len(vertices) >= 3, (0, 210, 255), 2)
                cv2.imshow(title, preview)
                key = cv2.waitKey(30) & 0xff
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 or key == 27:
                    raise RuntimeError("Target selection cancelled")
                if key in (10, 13) and len(vertices) >= 3:
                    return mask_from_polygon(depth_mm, vertices)
        finally:
            cv2.destroyWindow(title)
    elif mode == "rectangle":
        x, y, rw, rh = map(int, cv2.selectROI("Select target (Enter confirms)", bgr, False, False))
        cv2.destroyWindow("Select target (Enter confirms)")
        if rw <= 0 or rh <= 0:
            raise RuntimeError("No target selected")
        region = np.zeros((h, w), bool)
        region[y:y+rh, x:x+rw] = True
    elif mode == "click":
        selected = []
        def callback(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected.append((x, y))
        cv2.namedWindow("Click target (Esc cancels)")
        cv2.setMouseCallback("Click target (Esc cancels)", callback)
        while not selected:
            cv2.imshow("Click target (Esc cancels)", bgr)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyWindow("Click target (Esc cancels)")
        if not selected:
            raise RuntimeError("No target selected")
        x, y = selected[0]
        region = mask_from_click(depth_mm, x, y, depth_tolerance_mm)
    else:
        raise ValueError("mode must be polygon, rectangle or click")
    # In a drawn box, take the nearest populated depth layer, then its largest island.
    # This separates foreground from a deeper table without picking a grasp center.
    valid = region & np.isfinite(depth_mm) & (depth_mm > 0)
    if mode == "rectangle":
        depth_values = np.asarray(depth_mm[valid], np.float32)
        if len(depth_values) < 30:
            raise RuntimeError("No valid target depth")
        nearest = float(np.percentile(depth_values, 10))
        valid &= np.abs(np.asarray(depth_mm, np.float32) - nearest) <= depth_tolerance_mm
        count, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), 8)
        if count < 2:
            raise RuntimeError("No valid target depth")
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        valid = labels == largest
    if valid.sum() < 30:
        raise RuntimeError("Target region has fewer than 30 valid pixels")
    return valid
