"""Task-specific insertion measurement from registered Wrist RGB/depth only.

The 82 mm top-to-tip distance is the fixed task specification (32 mm handle +
50 mm pin). No model, object/site pose, contact, or simulator-success APIs are used.
This estimates relative surface height, not whether the pin actually enters the
bore or is still held; those conditions require independent visual qualification.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage


def _surface(mask, depth, *, vertical_fov_deg, min_pixels=64, plane_band_m=.0015, min_plane_fraction=.5):
    """Find an eroded, connected, dominant nearest-depth colored surface."""
    mask = ndimage.binary_erosion(mask, iterations=2, border_value=0)
    labels, count = ndimage.label(mask)
    if not count:
        return None, "No usable connected color surface"
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(np.argmax(sizes))
    component_pixels = int(component.sum())
    values = depth[component]
    if component_pixels < min_pixels:
        return None, "Too few eroded color-surface pixels"
    near_depth = float(np.quantile(values, .05))
    top = component & (depth >= near_depth - plane_band_m) & (depth <= near_depth + plane_band_m)
    top = ndimage.binary_erosion(top, iterations=1, border_value=0)
    top_labels, top_count = ndimage.label(top)
    if not top_count:
        return None, "No connected top-plane patch"
    top_sizes = np.bincount(top_labels.ravel())
    top_sizes[0] = 0
    top = top_labels == int(np.argmax(top_sizes))
    plane_pixels = int(top.sum())
    fraction = plane_pixels / component_pixels
    if plane_pixels < min_pixels or fraction < min_plane_fraction:
        return None, "Nearest surface is too small or not a dominant flat top"
    ys, xs = np.nonzero(top)
    if min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) < 6:
        return None, "Top-plane patch is too thin"
    values = depth[top]
    q05, median, q95 = (float(v) for v in np.quantile(values, [.05, .5, .95]))
    spread = q95 - q05
    if spread > plane_band_m:
        return None, "Top surface has excessive depth spread"
    # A narrow SIDE strip can also have a small scalar depth spread. Deproject
    # registered optical-axis depths and test its fitted 3D plane orientation.
    # Square camera pixels imply fx == fy, derived from vertical FOV and height.
    height, width = depth.shape
    focal = height / (2. * math.tan(math.radians(vertical_fov_deg) / 2.))
    points = np.column_stack(((xs + .5 - width / 2.) * values / focal,
                              (ys + .5 - height / 2.) * values / focal, values))
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None, "Top-plane fit failed"
    if eigenvalues[1] <= 1e-12:
        return None, "Top-plane patch has insufficient two-dimensional support"
    normal = eigenvectors[:, 0]
    normal_tilt = math.degrees(math.acos(float(np.clip(abs(normal[2]), 0., 1.))))
    plane_rms = math.sqrt(max(0., float(eigenvalues[0])))
    if normal_tilt > 10.:
        return None, "Colored surface is a side or tilted plane, not an inspectable top"
    if plane_rms > .0005:
        return None, "Top-plane residual exceeds 0.5 mm"
    return {"pixels": plane_pixels, "component_pixels": component_pixels,
            "plane_fraction": round(fraction, 4), "depth_m": median,
            "spread_m": spread, "q05_m": q05, "q95_m": q95,
            "normal_tilt_deg": normal_tilt, "plane_rms_m": plane_rms}, None


def estimate_insertion(obs, *, uncertainty_m=.003, minimum_insertion_m=.025, max_tilt_deg=1.0):
    """Return a bounded JSON-safe measurement, or an explicit uncertain result.

    ``qualified`` means only that the image-derived lower bound meets the depth
    requirement. It never proves bore alignment, source/destination identity, or
    continued grasp. Missing or unsuitable evidence cannot qualify.
    """
    result = {"quality": "uncertain", "qualified": False, "reason": None,
              "estimate_m": None, "lower_bound_m": None, "uncertainty_m": .003,
              "camera_tilt_deg": None, "vertical_fov_deg": None, "top_gap_m": None, "orange": None, "blue": None}
    def unavailable(reason):
        result["reason"] = reason
        return result
    try:
        if not isinstance(obs, dict):
            return unavailable("Observation must contain registered image evidence")
        if not all(math.isfinite(float(v)) for v in (uncertainty_m, minimum_insertion_m, max_tilt_deg)):
            return unavailable("Invalid measurement limits")
        if uncertainty_m < .003 or minimum_insertion_m < .025 or not 0 <= max_tilt_deg <= 1.0:
            return unavailable("Measurement limits must retain the 3 mm margin and 1 degree view bound")
        result["uncertainty_m"] = float(uncertainty_m)
        image = np.asarray(obs.get("wrist"))
        depth = np.asarray(obs.get("wrist_depth_m"), dtype=float)
        if image.ndim != 3 or image.shape[2] != 3 or depth.ndim != 2 or image.shape[:2] != depth.shape:
            return unavailable("Registered Wrist RGB and raw metric depth are required")
        if min(depth.shape) < 16 or image.dtype != np.uint8:
            return unavailable("Unsupported Wrist RGB dimensions or pixel type")
        sightline = np.asarray((obs.get("wrist_camera_axes") or {}).get("sightline"), dtype=float)
        if sightline.shape != (3,) or not np.isfinite(sightline).all() or np.linalg.norm(sightline) <= 0:
            return unavailable("Measured Wrist-camera sightline is required")
        sightline = sightline / np.linalg.norm(sightline)
        tilt = math.degrees(math.acos(float(np.clip(-sightline[2], -1., 1.))))
        result["camera_tilt_deg"] = tilt
        if tilt > max_tilt_deg:
            return unavailable("Wrist camera is not sufficiently vertical for a surface-height measurement")
        calibration = obs.get("wrist_depth_calibration") or {}
        # Legacy observations from this fixed task use the original 80-degree
        # Wrist lens. New observations carry its measured camera calibration.
        vertical_fov_deg = float(calibration.get("vertical_fov_deg", 80.))
        if not math.isfinite(vertical_fov_deg) or not 0 < vertical_fov_deg < 180:
            return unavailable("Invalid Wrist-camera vertical field of view")
        result["vertical_fov_deg"] = vertical_fov_deg
        near, far = float(calibration.get("near_m", 0.)), float(calibration.get("far_m", .3))
        if not math.isfinite(near) or not math.isfinite(far) or not 0 <= near < far:
            return unavailable("Invalid Wrist-depth range")
        valid = np.isfinite(depth) & (depth > max(0., near)) & (depth <= far)
        rgb = image.astype(np.float64)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        # Broad known-color separation tolerates lighting while excluding gray/white
        # robot surfaces. Erosion removes mixed-color and mixed-depth boundary pixels.
        orange_mask = valid & (red >= 60) & (green >= 15) & (red > 1.5 * green) & (green > 1.3 * blue) & (red - blue >= 40)
        blue_mask = valid & (blue >= 45) & (blue > 1.4 * red) & (blue > 1.1 * green) & (blue - red >= 25)
        for name, mask in (("orange", orange_mask), ("blue", blue_mask)):
            surface, error = _surface(mask, depth, vertical_fov_deg=vertical_fov_deg)
            result[name] = surface
            if error:
                return unavailable(name.capitalize() + " top: " + error)
        gap = result["blue"]["depth_m"] - result["orange"]["depth_m"]
        estimate = .082 - gap
        result.update(top_gap_m=gap, estimate_m=estimate, lower_bound_m=estimate - uncertainty_m)
        if gap < .032 - uncertainty_m:
            return unavailable("Observed top heights are inconsistent with an upright plug above the socket")
        qualified = result["lower_bound_m"] >= minimum_insertion_m
        result.update(quality="measured", qualified=bool(qualified),
                      reason="Image-derived depth lower bound meets 25 mm" if qualified else
                             "Image-derived depth lower bound is below 25 mm")
        return result
    except (TypeError, ValueError, IndexError, OverflowError):
        return unavailable("Malformed or unsupported RGB/depth measurement inputs")
