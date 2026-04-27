"""Unified iris pipeline: preprocessing, segmentation, encoding, matching, evaluation."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from iris.config import AppConfig
from iris.errors import ConfigError, InputDataError, IrisPipelineError

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Stage2Params:
    grayscale: bool
    normalize_histogram: bool
    gaussian_blur_kernel: int
    iris_threshold_divisor_xi: float
    pupil_threshold_divisor_xp: float
    save_debug_images: bool


@dataclass(frozen=True)
class Stage2Artifacts:
    gray: np.ndarray
    processed: np.ndarray
    iris_binary: np.ndarray
    pupil_binary: np.ndarray
    iris_threshold: int
    pupil_threshold: int
    mean_intensity: float


@dataclass(frozen=True)
class Stage3Params:
    morph_kernel_size: int
    min_component_area: int
    save_debug_images: bool
    hough_min_radius: int
    hough_max_radius: int


@dataclass(frozen=True)
class PupilSegmentationResult:
    status: str
    center_x: int
    center_y: int
    radius: int
    component: np.ndarray | None
    error_message: str | None = None


@dataclass(frozen=True)
class Stage4Params:
    iris_min_radius_scale: float
    iris_max_radius_scale: float
    radial_smoothing_window: int
    save_debug_images: bool


@dataclass(frozen=True)
class IrisSegmentationResult:
    status: str
    center_x: int
    center_y: int
    pupil_radius: int
    iris_radius: int
    error_message: str | None = None


@dataclass(frozen=True)
class Stage5Params:
    radial_samples_per_band: int
    upper_angle_exclusion_deg: float
    lower_angle_exclusion_deg: float
    radial_bands: int
    angular_samples: int
    save_debug_images: bool


@dataclass(frozen=True)
class IrisNormalizationResult:
    status: str
    normalized: np.ndarray | None
    error_message: str | None = None


@dataclass(frozen=True)
class Stage6Params:
    radial_bands: int
    angular_samples: int
    radial_samples_per_band: int
    gabor_frequency: float
    response_reliability_threshold: float
    save_debug_images: bool


@dataclass(frozen=True)
class Stage7Params:
    verify_threshold: float
    max_angular_shift: int


@dataclass(frozen=True)
class IrisTemplate:
    image_name: str
    subject_id: str
    code: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class Stage8Params:
    threshold_steps: int


def _collect_input_images(input_dir: Path) -> list[Path]:
    images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not images:
        raise InputDataError(f"No supported image files in input directory: {input_dir}")
    return images


def _safe_image_id(input_dir: Path, image_path: Path) -> str:
    relative = image_path.relative_to(input_dir).with_suffix("")
    safe = "_".join(relative.parts)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or image_path.stem


def _subject_id_from_input_path(input_dir: Path, image_path: Path) -> str:
    stem = image_path.stem
    generic_tokens = {"left", "right", "l", "r", "images", "iris", "dataset"}
    relative_parts = image_path.relative_to(input_dir).parts
    if len(relative_parts) >= 3:
        side = relative_parts[-2].lower()
        owner = relative_parts[-3]
        if side in {"left", "right"} and owner.lower() not in generic_tokens:
            return f"{owner}_{side}"
    for pattern in (r"^(.*)_\d+$", r"^(.*\d)[Rr]\d+$", r"^([A-Za-z][A-Za-z0-9_-]*[A-Za-z])(\d+)$"):
        match = re.match(pattern, stem)
        if match:
            candidate = match.group(1)
            if candidate.rstrip("_").lower() not in generic_tokens:
                return candidate
    if len(relative_parts) > 1:
        parent = relative_parts[-2]
        if parent.lower() not in generic_tokens:
            return parent
    return stem


def _extract_stage2_params(app_config: AppConfig) -> Stage2Params:
    preprocessing = app_config.raw.get("preprocessing", {})
    segmentation = app_config.raw.get("segmentation", {})
    runtime = app_config.raw.get("runtime", {})
    grayscale = preprocessing.get("grayscale", True)
    normalize_histogram = preprocessing.get("normalize_histogram", False)
    gaussian_blur_kernel = preprocessing.get("gaussian_blur_kernel", 3)
    xi = segmentation.get("iris_threshold_divisor_xi")
    xp = segmentation.get("pupil_threshold_divisor_xp")
    save_debug_images = runtime.get("save_debug_images", True)
    if not isinstance(grayscale, bool):
        raise ConfigError("'preprocessing.grayscale' must be a boolean.")
    if not isinstance(normalize_histogram, bool):
        raise ConfigError("'preprocessing.normalize_histogram' must be a boolean.")
    if not isinstance(gaussian_blur_kernel, int) or gaussian_blur_kernel < 1 or gaussian_blur_kernel % 2 == 0:
        raise ConfigError("'preprocessing.gaussian_blur_kernel' must be odd integer >= 1.")
    if not isinstance(xi, (int, float)) or float(xi) <= 0:
        raise ConfigError("'segmentation.iris_threshold_divisor_xi' must be > 0.")
    if not isinstance(xp, (int, float)) or float(xp) <= 0:
        raise ConfigError("'segmentation.pupil_threshold_divisor_xp' must be > 0.")
    if not isinstance(save_debug_images, bool):
        raise ConfigError("'runtime.save_debug_images' must be boolean.")
    return Stage2Params(grayscale, normalize_histogram, gaussian_blur_kernel, float(xi), float(xp), save_debug_images)


def _compute_stage2_artifacts(image_bgr: np.ndarray, params: Stage2Params) -> Stage2Artifacts:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if params.grayscale else image_bgr[:, :, 0]
    processed = gray.copy()
    if params.normalize_histogram:
        processed = cv2.equalizeHist(processed)
    if params.gaussian_blur_kernel > 1:
        processed = cv2.GaussianBlur(processed, (params.gaussian_blur_kernel, params.gaussian_blur_kernel), 0)
    mean_intensity = float(np.mean(processed))
    iris_threshold = int(np.clip(mean_intensity / params.iris_threshold_divisor_xi, 0, 255))
    pupil_threshold = int(np.clip(mean_intensity / params.pupil_threshold_divisor_xp, 0, 255))
    _, iris_binary = cv2.threshold(processed, iris_threshold, 255, cv2.THRESH_BINARY_INV)
    _, pupil_binary = cv2.threshold(processed, pupil_threshold, 255, cv2.THRESH_BINARY_INV)
    return Stage2Artifacts(gray, processed, iris_binary, pupil_binary, iris_threshold, pupil_threshold, mean_intensity)


def _save_stage2_debug(output_dir: Path, image_name: str, artifacts: Stage2Artifacts) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    base = Path(image_name).stem
    writes = [
        (debug_dir / f"{base}_gray.png", artifacts.gray),
        (debug_dir / f"{base}_processed.png", artifacts.processed),
        (debug_dir / f"{base}_iris_binary.png", artifacts.iris_binary),
        (debug_dir / f"{base}_pupil_binary.png", artifacts.pupil_binary),
    ]
    for path, img in writes:
        if not cv2.imwrite(str(path), img):
            raise IrisPipelineError(f"Failed to save debug image: {path}")


def run_stage2_preprocessing(app_config: AppConfig) -> None:
    params = _extract_stage2_params(app_config)
    images = _collect_input_images(app_config.input_dir)
    LOGGER.info("Stage 2 started for %d image(s).", len(images))
    for image_path in images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise InputDataError(f"Failed to load image file: {image_path}")
        artifacts = _compute_stage2_artifacts(image_bgr, params)
        LOGGER.info(
            "Processed %s | mean=%.2f | PI=%d | PP=%d",
            image_path.name,
            artifacts.mean_intensity,
            artifacts.iris_threshold,
            artifacts.pupil_threshold,
        )
        if params.save_debug_images:
            _save_stage2_debug(app_config.output_dir, image_path.name, artifacts)
    LOGGER.info("Stage 2 finished successfully.")


def _extract_stage3_params(app_config: AppConfig) -> Stage3Params:
    segmentation = app_config.raw.get("segmentation", {})
    runtime = app_config.raw.get("runtime", {})
    morph_kernel_size = segmentation.get("morph_kernel_size", 3)
    min_component_area = segmentation.get("min_component_area", 150)
    hough_min_radius = segmentation.get("pupil_hough_min_radius", 20)
    hough_max_radius = segmentation.get("pupil_hough_max_radius", 220)
    save_debug_images = runtime.get("save_debug_images", True)
    if not isinstance(morph_kernel_size, int) or morph_kernel_size < 1 or morph_kernel_size % 2 == 0:
        raise ConfigError("'segmentation.morph_kernel_size' must be odd integer >= 1.")
    if not isinstance(min_component_area, int) or min_component_area < 1:
        raise ConfigError("'segmentation.min_component_area' must be integer >= 1.")
    if not isinstance(hough_min_radius, int) or hough_min_radius < 5:
        raise ConfigError("'segmentation.pupil_hough_min_radius' must be integer >= 5.")
    if not isinstance(hough_max_radius, int) or hough_max_radius <= hough_min_radius:
        raise ConfigError("'segmentation.pupil_hough_max_radius' must be > pupil_hough_min_radius.")
    if not isinstance(save_debug_images, bool):
        raise ConfigError("'runtime.save_debug_images' must be boolean.")
    return Stage3Params(morph_kernel_size, min_component_area, save_debug_images, hough_min_radius, hough_max_radius)


def _morph_cleanup(mask: np.ndarray, params: Stage3Params) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.morph_kernel_size, params.morph_kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)


def _focus_center(mask: np.ndarray, keep_ratio: float = 0.7) -> np.ndarray:
    h, w = mask.shape
    kh, kw = int(h * keep_ratio), int(w * keep_ratio)
    y0, x0 = (h - kh) // 2, (w - kw) // 2
    focused = np.zeros_like(mask, dtype=np.uint8)
    focused[y0 : y0 + kh, x0 : x0 + kw] = mask[y0 : y0 + kh, x0 : x0 + kw]
    return focused


def _component_circularity(component: np.ndarray, area: float) -> float:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    perimeter = float(cv2.arcLength(contours[0], closed=True))
    if perimeter <= 0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def _select_pupil(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise IrisPipelineError("Pupil segmentation failed: no foreground component found.")
    best_label, best_score = -1, float("-inf")
    center = np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
    diag = float(np.hypot(mask.shape[1], mask.shape[0]))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = np.zeros_like(mask, dtype=np.uint8)
        component[labels == label] = 255
        ys, xs = np.where(component > 0)
        if len(xs) == 0:
            continue
        centroid = np.array([float(np.mean(xs)), float(np.mean(ys))])
        dist_score = 1.0 - float(np.linalg.norm(centroid - center) / max(diag, 1.0))
        circ_score = _component_circularity(component, float(area))
        area_score = float(area) / float(mask.shape[0] * mask.shape[1])
        score = (1.5 * circ_score) + (1.0 * dist_score) + (0.2 * area_score)
        if score > best_score:
            best_score = score
            best_label = label
    if best_label < 0:
        raise IrisPipelineError("Pupil segmentation failed: no component above minimum area.")
    component = np.zeros_like(mask, dtype=np.uint8)
    component[labels == best_label] = 255
    return component


def _pupil_center_radius(component: np.ndarray) -> tuple[int, int, int]:
    horizontal = np.sum(component > 0, axis=1)
    vertical = np.sum(component > 0, axis=0)
    cy = int(np.argmax(horizontal))
    cx = int(np.argmax(vertical))
    area = float(np.count_nonzero(component))
    radius = max(int(round(np.sqrt(area / np.pi))), 1)
    return cx, cy, radius


def _hough_fallback(gray: np.ndarray, params: Stage3Params) -> tuple[int, int, int] | None:
    max_dim = max(gray.shape)
    scale = 1200.0 / float(max_dim) if max_dim > 1200 else 1.0
    resized = cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    blurred = cv2.medianBlur(resized, 5)
    min_r = max(5, int(round(params.hough_min_radius * scale)))
    max_r = max(min_r + 2, int(round(params.hough_max_radius * scale)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, resized.shape[0] // 8),
        param1=120,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    center = np.array([resized.shape[1] / 2.0, resized.shape[0] / 2.0], dtype=np.float32)
    best_score = float("-inf")
    best: tuple[int, int, int] | None = None
    for circle in circles[0]:
        cx, cy, radius = int(round(circle[0])), int(round(circle[1])), int(round(circle[2]))
        dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center))
        centrality = 1.0 - (dist / max(np.hypot(*center), 1.0))
        radius_score = min(radius / max(max_r, 1), 1.0)
        score = (1.2 * centrality) + (0.5 * radius_score)
        if score > best_score:
            best_score = score
            best = (cx, cy, radius)
    if best is None:
        return None
    if scale < 1.0:
        inv = 1.0 / scale
        return int(round(best[0] * inv)), int(round(best[1] * inv)), int(round(best[2] * inv))
    return best


def _pupil_circle_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = dist <= radius
    ring = (dist > radius * 1.10) & (dist <= radius * 1.60)
    if int(np.count_nonzero(inner)) < 50 or int(np.count_nonzero(ring)) < 50:
        return float("-inf")
    inner_mean = float(np.mean(gray[inner]))
    ring_mean = float(np.mean(gray[ring]))
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    dist_from_center = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center))
    centrality = 1.0 - (dist_from_center / max(float(np.hypot(*center)), 1.0))
    return (ring_mean - inner_mean) + (25.0 * centrality)


def _percentile_pupil_candidate(gray: np.ndarray, params: Stage3Params) -> tuple[int, int, int, np.ndarray] | None:
    h, w = gray.shape
    x0, x1 = int(w * 0.15), int(w * 0.85)
    y0, y1 = int(h * 0.15), int(h * 0.85)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    best_score = float("-inf")
    best_result: tuple[int, int, int, np.ndarray] | None = None
    max_area = int(0.08 * h * w)
    max_radius = max(20, min(params.hough_max_radius, int(round(min(gray.shape) * 0.22))))
    for percentile in (1, 2, 3, 4, 5, 6, 8, 10):
        threshold = float(np.percentile(roi, percentile))
        mask = (gray <= threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(200, params.min_component_area) or area > max_area:
                continue
            x, y, cw, ch = [int(v) for v in stats[label, :4]]
            if x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2:
                continue
            component = np.zeros_like(mask, dtype=np.uint8)
            component[labels == label] = 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < 20 or radius > max_radius:
                continue
            perimeter = float(cv2.arcLength(contour, closed=True))
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
            area_ratio = float(area) / max(math.pi * radius * radius, 1.0)
            score = _pupil_circle_score(gray, cx, cy, radius) + (20.0 * circularity) + (20.0 * area_ratio)
            if score > best_score:
                best_score = score
                best_result = (int(round(cx)), int(round(cy)), int(round(radius)), component)
    return best_result


def segment_pupil_from_stage2(artifacts: Stage2Artifacts, params: Stage3Params) -> PupilSegmentationResult:
    cleaned = _morph_cleanup(artifacts.pupil_binary, params)
    try:
        focused = _focus_center(cleaned, keep_ratio=0.7)
        try:
            component = _select_pupil(focused, params.min_component_area)
        except IrisPipelineError:
            component = _select_pupil(cleaned, max(20, params.min_component_area // 4))
        cx, cy, radius = _pupil_center_radius(component)
        max_reasonable_radius = max(20, int(round(min(artifacts.processed.shape) * 0.25)))
        if radius < 20 or radius > max_reasonable_radius:
            raise IrisPipelineError(f"Estimated pupil radius out of range ({radius}px).")
        return PupilSegmentationResult("ok", cx, cy, radius, component)
    except IrisPipelineError as exc:
        percentile_candidate = _percentile_pupil_candidate(artifacts.processed, params)
        if percentile_candidate is not None:
            cx, cy, radius, component = percentile_candidate
            return PupilSegmentationResult("ok", cx, cy, radius, component, "fallback:percentile")
        circle = _hough_fallback(artifacts.processed, params)
        if circle is None:
            return PupilSegmentationResult("failed", -1, -1, -1, None, str(exc))
        cx, cy, radius = circle
        component = np.zeros_like(artifacts.processed, dtype=np.uint8)
        cv2.circle(component, (cx, cy), radius, 255, -1)
        return PupilSegmentationResult("ok", cx, cy, radius, component, "fallback:hough")


def _save_stage3_debug(output_dir: Path, image_name: str, component: np.ndarray, cx: int, cy: int, radius: int) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    base = Path(image_name).stem
    overlay = cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
    cv2.circle(overlay, (cx, cy), radius, (0, 255, 0), 2)
    cv2.circle(overlay, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(str(debug_dir / f"{base}_pupil_cleaned.png"), component)
    cv2.imwrite(str(debug_dir / f"{base}_pupil_circle_overlay.png"), overlay)


def run_stage3_pupil_segmentation(app_config: AppConfig) -> None:
    stage2_params = _extract_stage2_params(app_config)
    stage3_params = _extract_stage3_params(app_config)
    images = _collect_input_images(app_config.input_dir)
    rows: list[dict[str, int | str]] = []
    LOGGER.info("Stage 3 started for %d image(s).", len(images))
    for image_path in images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise InputDataError(f"Failed to load image file: {image_path}")
        artifacts = _compute_stage2_artifacts(image_bgr, stage2_params)
        result = segment_pupil_from_stage2(artifacts, stage3_params)
        if result.status == "ok":
            LOGGER.info("Segmented pupil %s | center=(%d,%d) | radius=%d", image_path.name, result.center_x, result.center_y, result.radius)
            rows.append({"image": image_path.name, "pupil_center_x": result.center_x, "pupil_center_y": result.center_y, "pupil_radius": result.radius, "status": "ok"})
            if stage3_params.save_debug_images and result.component is not None:
                _save_stage3_debug(app_config.output_dir, image_path.name, result.component, result.center_x, result.center_y, result.radius)
        else:
            LOGGER.warning("Pupil segmentation skipped for %s: %s", image_path.name, result.error_message)
            rows.append({"image": image_path.name, "pupil_center_x": -1, "pupil_center_y": -1, "pupil_radius": -1, "status": "failed"})
    path = app_config.output_dir / "pupil_segmentation.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "pupil_center_x", "pupil_center_y", "pupil_radius", "status"])
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Stage 3 finished successfully. Saved: %s", path)


def _extract_stage4_params(app_config: AppConfig) -> Stage4Params:
    segmentation = app_config.raw.get("segmentation", {})
    runtime = app_config.raw.get("runtime", {})
    min_scale = segmentation.get("iris_min_radius_scale", 1.8)
    max_scale = segmentation.get("iris_max_radius_scale", 4.0)
    smoothing = segmentation.get("radial_smoothing_window", 9)
    save_debug_images = runtime.get("save_debug_images", True)
    if not isinstance(min_scale, (int, float)) or float(min_scale) <= 1.0:
        raise ConfigError("'segmentation.iris_min_radius_scale' must be > 1.0.")
    if not isinstance(max_scale, (int, float)) or float(max_scale) <= float(min_scale):
        raise ConfigError("'segmentation.iris_max_radius_scale' must be > iris_min_radius_scale.")
    if not isinstance(smoothing, int) or smoothing < 3 or smoothing % 2 == 0:
        raise ConfigError("'segmentation.radial_smoothing_window' must be odd integer >= 3.")
    if not isinstance(save_debug_images, bool):
        raise ConfigError("'runtime.save_debug_images' must be boolean.")
    return Stage4Params(float(min_scale), float(max_scale), smoothing, save_debug_images)


def _radial_profile(gray: np.ndarray, cx: int, cy: int, radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    means: list[float] = []
    h, w = gray.shape
    for radius in radii:
        xs = np.round(cx + radius * np.cos(angles)).astype(np.int32)
        ys = np.round(cy + radius * np.sin(angles)).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        means.append(float(np.mean(gray[ys[valid], xs[valid]])) if np.any(valid) else 0.0)
    return np.asarray(means, dtype=np.float32)


def _estimate_iris_radius(gray: np.ndarray, cx: int, cy: int, pupil_radius: int, params: Stage4Params) -> int:
    r_min = int(round(max(pupil_radius * params.iris_min_radius_scale, pupil_radius + 10)))
    r_max = int(round(min(pupil_radius * params.iris_max_radius_scale, min(gray.shape) // 2 - 1)))
    if r_max <= r_min + 5:
        raise IrisPipelineError("Iris radius search range is too narrow.")
    radii = np.arange(r_min, r_max + 1, dtype=np.int32)
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False, dtype=np.float32)
    profile = _radial_profile(gray, cx, cy, radii, angles)
    smooth = np.convolve(profile, np.ones(params.radial_smoothing_window, dtype=np.float32) / float(params.radial_smoothing_window), mode="same")
    gradient = np.gradient(smooth)
    iris_radius = int(radii[int(np.argmax(gradient))])
    if iris_radius <= pupil_radius + 5:
        raise IrisPipelineError("Estimated iris radius is not larger than pupil radius.")
    return iris_radius


def segment_iris_from_stage2(artifacts: Stage2Artifacts, stage3_params: Stage3Params, stage4_params: Stage4Params) -> IrisSegmentationResult:
    pupil = segment_pupil_from_stage2(artifacts, stage3_params)
    if pupil.status != "ok":
        return IrisSegmentationResult("failed", -1, -1, -1, -1, "missing pupil")
    try:
        iris_radius = _estimate_iris_radius(artifacts.processed, pupil.center_x, pupil.center_y, pupil.radius, stage4_params)
        return IrisSegmentationResult("ok", pupil.center_x, pupil.center_y, pupil.radius, iris_radius)
    except IrisPipelineError as exc:
        return IrisSegmentationResult("failed", pupil.center_x, pupil.center_y, pupil.radius, -1, str(exc))


def _save_stage4_debug(output_dir: Path, image_name: str, gray: np.ndarray, cx: int, cy: int, pr: int, ir: int) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    base = Path(image_name).stem
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(overlay, (cx, cy), pr, (0, 255, 0), 2)
    cv2.circle(overlay, (cx, cy), ir, (255, 0, 0), 2)
    cv2.circle(overlay, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(str(debug_dir / f"{base}_iris_circle_overlay.png"), overlay)


def run_stage4_iris_segmentation(app_config: AppConfig) -> None:
    stage2_params = _extract_stage2_params(app_config)
    stage3_params = _extract_stage3_params(app_config)
    stage4_params = _extract_stage4_params(app_config)
    images = _collect_input_images(app_config.input_dir)
    rows: list[dict[str, int | str]] = []
    LOGGER.info("Stage 4 started for %d image(s).", len(images))
    for image_path in images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise InputDataError(f"Failed to load image file: {image_path}")
        artifacts = _compute_stage2_artifacts(image_bgr, stage2_params)
        result = segment_iris_from_stage2(artifacts, stage3_params, stage4_params)
        if result.status == "ok":
            LOGGER.info("Segmented iris %s | center=(%d,%d) | pupil_r=%d | iris_r=%d", image_path.name, result.center_x, result.center_y, result.pupil_radius, result.iris_radius)
            rows.append({"image": image_path.name, "pupil_center_x": result.center_x, "pupil_center_y": result.center_y, "pupil_radius": result.pupil_radius, "iris_radius": result.iris_radius, "status": "ok"})
            if stage4_params.save_debug_images:
                _save_stage4_debug(app_config.output_dir, image_path.name, artifacts.processed, result.center_x, result.center_y, result.pupil_radius, result.iris_radius)
        else:
            LOGGER.warning("Iris segmentation skipped for %s: %s.", image_path.name, result.error_message)
            rows.append({"image": image_path.name, "pupil_center_x": result.center_x, "pupil_center_y": result.center_y, "pupil_radius": result.pupil_radius, "iris_radius": -1, "status": "failed"})
    path = app_config.output_dir / "iris_segmentation.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "pupil_center_x", "pupil_center_y", "pupil_radius", "iris_radius", "status"])
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Stage 4 finished successfully. Saved: %s", path)


def _extract_stage5_params(app_config: AppConfig) -> Stage5Params:
    normalization = app_config.raw.get("normalization", {})
    encoding = app_config.raw.get("encoding", {})
    runtime = app_config.raw.get("runtime", {})
    radial_samples_per_band = normalization.get("radial_samples_per_band", 16)
    upper_excl = normalization.get("upper_angle_exclusion_deg", 35.0)
    lower_excl = normalization.get("lower_angle_exclusion_deg", 20.0)
    radial_bands = encoding.get("radial_bands", 8)
    angular_samples = encoding.get("angular_samples", 128)
    save_debug_images = runtime.get("save_debug_images", True)
    if not isinstance(radial_samples_per_band, int) or radial_samples_per_band < 4:
        raise ConfigError("'normalization.radial_samples_per_band' must be integer >= 4.")
    if not isinstance(upper_excl, (int, float)) or not 0 <= float(upper_excl) < 85:
        raise ConfigError("'normalization.upper_angle_exclusion_deg' must be in [0, 85).")
    if not isinstance(lower_excl, (int, float)) or not 0 <= float(lower_excl) < 85:
        raise ConfigError("'normalization.lower_angle_exclusion_deg' must be in [0, 85).")
    if float(upper_excl) + float(lower_excl) >= 150:
        raise ConfigError("Too much angular exclusion: upper + lower must be < 150 deg.")
    if not isinstance(radial_bands, int) or radial_bands < 1:
        raise ConfigError("'encoding.radial_bands' must be integer >= 1.")
    if not isinstance(angular_samples, int) or angular_samples < 32:
        raise ConfigError("'encoding.angular_samples' must be integer >= 32.")
    if not isinstance(save_debug_images, bool):
        raise ConfigError("'runtime.save_debug_images' must be boolean.")
    return Stage5Params(radial_samples_per_band, float(upper_excl), float(lower_excl), radial_bands, angular_samples, save_debug_images)


def _build_theta_grid(params: Stage5Params) -> np.ndarray:
    bottom_start = 90.0 - params.lower_angle_exclusion_deg
    bottom_end = 90.0 + params.lower_angle_exclusion_deg
    top_start = 270.0 - params.upper_angle_exclusion_deg
    top_end = 270.0 + params.upper_angle_exclusion_deg
    left_arc_start = bottom_end
    left_arc_end = top_start
    right_arc_start = top_end
    right_arc_end = 360.0 + bottom_start
    left_span = left_arc_end - left_arc_start
    right_span = right_arc_end - right_arc_start
    total_span = left_span + right_span
    if left_span <= 0 or right_span <= 0 or total_span <= 0:
        raise IrisPipelineError("Invalid angular range after exclusion setup.")
    left_samples = max(1, int(round(params.angular_samples * (left_span / total_span))))
    right_samples = params.angular_samples - left_samples
    if right_samples <= 0:
        right_samples = 1
        left_samples = params.angular_samples - 1
    left = np.linspace(np.deg2rad(left_arc_start), np.deg2rad(left_arc_end), left_samples, endpoint=False)
    right = np.linspace(np.deg2rad(right_arc_start), np.deg2rad(right_arc_end), right_samples, endpoint=False)
    return np.concatenate([left, right]).astype(np.float32)


def _normalize_iris(gray: np.ndarray, cx: int, cy: int, pupil_radius: int, iris_radius: int, params: Stage5Params) -> np.ndarray:
    total_radial = params.radial_bands * params.radial_samples_per_band
    radial_steps = np.linspace(0.0, 1.0, total_radial, dtype=np.float32)
    theta = _build_theta_grid(params).astype(np.float32)
    radii = pupil_radius + radial_steps[:, None] * (iris_radius - pupil_radius)
    xs = cx + radii * np.cos(theta)[None, :]
    ys = cy + radii * np.sin(theta)[None, :]
    map_x = np.clip(xs, 0, gray.shape[1] - 1).astype(np.float32)
    map_y = np.clip(ys, 0, gray.shape[0] - 1).astype(np.float32)
    return cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def normalize_iris_from_stage2(artifacts: Stage2Artifacts, stage3_params: Stage3Params, stage4_params: Stage4Params, stage5_params: Stage5Params) -> IrisNormalizationResult:
    iris = segment_iris_from_stage2(artifacts, stage3_params, stage4_params)
    if iris.status != "ok":
        return IrisNormalizationResult("failed", None, iris.error_message)
    normalized = _normalize_iris(artifacts.processed, iris.center_x, iris.center_y, iris.pupil_radius, iris.iris_radius, stage5_params)
    return IrisNormalizationResult("ok", normalized)


def _save_stage5_outputs(output_dir: Path, image_stem: str, normalized: np.ndarray, params: Stage5Params) -> tuple[Path, Path]:
    normalized_dir = output_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    png_path = normalized_dir / f"{image_stem}_normalized.png"
    npy_path = normalized_dir / f"{image_stem}_normalized.npy"
    if not cv2.imwrite(str(png_path), normalized):
        raise IrisPipelineError(f"Failed to save normalized iris image: {png_path}")
    np.save(npy_path, normalized)
    if params.save_debug_images:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        for band_idx in range(1, params.radial_bands):
            y = band_idx * params.radial_samples_per_band
            cv2.line(overlay, (0, y), (overlay.shape[1] - 1, y), (255, 255, 255), 1)
        cv2.imwrite(str(debug_dir / f"{image_stem}_normalized_bands.png"), overlay)
    return png_path, npy_path


def run_stage5_normalization(app_config: AppConfig) -> None:
    stage2_params = _extract_stage2_params(app_config)
    stage3_params = _extract_stage3_params(app_config)
    stage4_params = _extract_stage4_params(app_config)
    stage5_params = _extract_stage5_params(app_config)
    images = _collect_input_images(app_config.input_dir)
    rows: list[dict[str, str | int]] = []
    LOGGER.info("Stage 5 started for %d image(s).", len(images))
    for image_path in images:
        image_id = _safe_image_id(app_config.input_dir, image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise InputDataError(f"Failed to load image file: {image_path}")
        artifacts = _compute_stage2_artifacts(image_bgr, stage2_params)
        result = normalize_iris_from_stage2(artifacts, stage3_params, stage4_params, stage5_params)
        if result.status != "ok" or result.normalized is None:
            LOGGER.warning("Normalization skipped for %s: %s", image_path.name, result.error_message)
            rows.append({"image": image_path.name, "normalized_png": "", "normalized_npy": "", "height": -1, "width": -1, "radial_bands": stage5_params.radial_bands, "angular_samples": stage5_params.angular_samples, "status": "failed"})
            continue
        png_path, npy_path = _save_stage5_outputs(app_config.output_dir, image_id, result.normalized, stage5_params)
        LOGGER.info("Normalized iris %s | size=%dx%d", image_path.name, result.normalized.shape[0], result.normalized.shape[1])
        rows.append({"image": image_path.name, "normalized_png": str(png_path), "normalized_npy": str(npy_path), "height": int(result.normalized.shape[0]), "width": int(result.normalized.shape[1]), "radial_bands": stage5_params.radial_bands, "angular_samples": stage5_params.angular_samples, "status": "ok"})
    path = app_config.output_dir / "normalized_iris.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "normalized_png", "normalized_npy", "height", "width", "radial_bands", "angular_samples", "status"])
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Stage 5 finished successfully. Saved: %s", path)


def _extract_stage6_params(app_config: AppConfig) -> Stage6Params:
    encoding = app_config.raw.get("encoding", {})
    normalization = app_config.raw.get("normalization", {})
    runtime = app_config.raw.get("runtime", {})
    radial_bands = encoding.get("radial_bands", 8)
    angular_samples = encoding.get("angular_samples", 128)
    gabor_frequency = encoding.get("gabor_frequency")
    radial_samples_per_band = normalization.get("radial_samples_per_band", 16)
    threshold = encoding.get("response_reliability_threshold", 0.015)
    save_debug_images = runtime.get("save_debug_images", True)
    if not isinstance(radial_bands, int) or radial_bands != 8:
        raise ConfigError("'encoding.radial_bands' must be exactly 8.")
    if not isinstance(angular_samples, int) or angular_samples != 128:
        raise ConfigError("'encoding.angular_samples' must be exactly 128.")
    if not isinstance(radial_samples_per_band, int) or radial_samples_per_band < 4:
        raise ConfigError("'normalization.radial_samples_per_band' must be integer >= 4.")
    if not isinstance(gabor_frequency, (int, float)) or float(gabor_frequency) <= 0:
        raise ConfigError("'encoding.gabor_frequency' must be > 0.")
    if not isinstance(threshold, (int, float)) or float(threshold) < 0:
        raise ConfigError("'encoding.response_reliability_threshold' must be >= 0.")
    if not isinstance(save_debug_images, bool):
        raise ConfigError("'runtime.save_debug_images' must be boolean.")
    return Stage6Params(radial_bands, angular_samples, radial_samples_per_band, float(gabor_frequency), float(threshold), save_debug_images)


def _gaussian_weights(length: int, sigma: float) -> np.ndarray:
    coords = np.arange(length, dtype=np.float32)
    center = (length - 1) / 2.0
    w = np.exp(-0.5 * ((coords - center) / max(sigma, 1e-6)) ** 2)
    return w / float(np.sum(w))


def _band_signals(normalized: np.ndarray, params: Stage6Params) -> np.ndarray:
    expected_h = params.radial_bands * params.radial_samples_per_band
    if normalized.shape != (expected_h, params.angular_samples):
        raise IrisPipelineError(f"Normalized image has unexpected shape {normalized.shape}.")
    weights = _gaussian_weights(params.radial_samples_per_band, sigma=params.radial_samples_per_band / 3.0)
    signals = np.zeros((params.radial_bands, params.angular_samples), dtype=np.float32)
    for b in range(params.radial_bands):
        y0 = b * params.radial_samples_per_band
        sl = normalized[y0 : y0 + params.radial_samples_per_band, :].astype(np.float32) / 255.0
        signals[b, :] = np.tensordot(weights, sl, axes=(0, 0))
    return signals


def _gabor_kernel_1d(frequency: float, angular_samples: int, half_size: int = 15) -> np.ndarray:
    sigma = 0.5 * np.pi * frequency
    if sigma <= 0:
        raise IrisPipelineError("Invalid Gabor sigma.")
    theta_step = (2.0 * np.pi) / float(angular_samples)
    x = np.arange(-half_size, half_size + 1, dtype=np.float32) * theta_step
    gaussian = np.exp(-(x**2) / (2.0 * sigma**2))
    carrier = np.exp(1j * 2.0 * np.pi * frequency * x)
    kernel = gaussian * carrier
    norm = float(np.sum(np.abs(kernel)))
    if norm <= 0:
        raise IrisPipelineError("Invalid Gabor kernel norm.")
    return kernel / norm


def _encode_signals(signals: np.ndarray, params: Stage6Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = _gabor_kernel_1d(params.gabor_frequency, params.angular_samples)
    code_bits = np.zeros((params.radial_bands, params.angular_samples, 2), dtype=np.uint8)
    mask_bits = np.zeros_like(code_bits)
    magnitude = np.zeros((params.radial_bands, params.angular_samples), dtype=np.float32)
    for b in range(params.radial_bands):
        pad = len(kernel) // 2
        response = np.convolve(np.pad(signals[b], (pad, pad), mode="wrap"), kernel, mode="valid")
        real, imag = np.real(response), np.imag(response)
        mag = np.sqrt(real**2 + imag**2).astype(np.float32)
        magnitude[b, :] = mag
        reliable = (mag >= params.response_reliability_threshold).astype(np.uint8)
        code_bits[b, :, 0] = (real >= 0).astype(np.uint8)
        code_bits[b, :, 1] = (imag >= 0).astype(np.uint8)
        mask_bits[b, :, 0] = reliable
        mask_bits[b, :, 1] = reliable
    return code_bits, mask_bits, magnitude


def _save_stage6(output_dir: Path, stem: str, code_bits: np.ndarray, mask_bits: np.ndarray, magnitude: np.ndarray, params: Stage6Params) -> tuple[Path, Path]:
    enc_dir = output_dir / "encoding"
    enc_dir.mkdir(parents=True, exist_ok=True)
    code_path = enc_dir / f"{stem}_iris_code.npy"
    mask_path = enc_dir / f"{stem}_iris_mask.npy"
    np.save(code_path, code_bits)
    np.save(mask_path, mask_bits)
    if params.save_debug_images:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        heatmap = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(debug_dir / f"{stem}_gabor_magnitude.png"), cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO))
    return code_path, mask_path


def run_stage6_encoding(app_config: AppConfig) -> None:
    stage2_params = _extract_stage2_params(app_config)
    stage3_params = _extract_stage3_params(app_config)
    stage4_params = _extract_stage4_params(app_config)
    stage5_params = _extract_stage5_params(app_config)
    stage6_params = _extract_stage6_params(app_config)
    images = _collect_input_images(app_config.input_dir)
    rows: list[dict[str, str | int | float]] = []
    LOGGER.info("Stage 6 started for %d image(s).", len(images))
    for image_path in images:
        image_id = _safe_image_id(app_config.input_dir, image_path)
        subject_id = _subject_id_from_input_path(app_config.input_dir, image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise InputDataError(f"Failed to load image file: {image_path}")
        artifacts = _compute_stage2_artifacts(image_bgr, stage2_params)
        norm = normalize_iris_from_stage2(artifacts, stage3_params, stage4_params, stage5_params)
        if norm.status != "ok" or norm.normalized is None:
            LOGGER.warning("Encoding skipped for %s: %s", image_path.name, norm.error_message)
            rows.append({"image": image_path.name, "subject_id": subject_id, "code_path": "", "mask_path": "", "code_bits_count": -1, "valid_bits_ratio": 0.0, "gabor_frequency": stage6_params.gabor_frequency, "status": "failed"})
            continue
        signals = _band_signals(norm.normalized, stage6_params)
        code_bits, mask_bits, magnitude = _encode_signals(signals, stage6_params)
        code_path, mask_path = _save_stage6(app_config.output_dir, image_id, code_bits, mask_bits, magnitude, stage6_params)
        valid_ratio = float(np.mean(mask_bits))
        LOGGER.info("Encoded iris %s | bits=%d | valid_ratio=%.3f", image_path.name, int(code_bits.size), valid_ratio)
        rows.append({"image": image_path.name, "subject_id": subject_id, "code_path": str(code_path), "mask_path": str(mask_path), "code_bits_count": int(code_bits.size), "valid_bits_ratio": valid_ratio, "gabor_frequency": stage6_params.gabor_frequency, "status": "ok"})
    path = app_config.output_dir / "iris_codes.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "subject_id", "code_path", "mask_path", "code_bits_count", "valid_bits_ratio", "gabor_frequency", "status"])
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Stage 6 finished successfully. Saved: %s", path)


def _extract_stage7_params(app_config: AppConfig) -> Stage7Params:
    matching = app_config.raw.get("matching", {})
    threshold = matching.get("verify_threshold", 0.34)
    max_shift = matching.get("max_angular_shift", 8)
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ConfigError("'matching.verify_threshold' must be in [0, 1].")
    if not isinstance(max_shift, int) or max_shift < 0:
        raise ConfigError("'matching.max_angular_shift' must be integer >= 0.")
    return Stage7Params(float(threshold), max_shift)


def _subject_id_from_name(stem: str) -> str:
    for pattern in (r"^(.*)_\d+$", r"^(.*\d)[Rr]\d+$", r"^([A-Za-z][A-Za-z0-9_-]*[A-Za-z])(\d+)$"):
        match = re.match(pattern, stem)
        if match:
            return match.group(1)
    return stem


def _load_templates(results_dir: Path) -> list[IrisTemplate]:
    manifest_path = results_dir / "iris_codes.csv"
    if not manifest_path.exists():
        raise IrisPipelineError(f"Missing encoding manifest: {manifest_path}")
    templates: list[IrisTemplate] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader if row.get("status") == "ok" and row.get("code_path") and row.get("mask_path")]
    if not rows:
        raise IrisPipelineError("No successful encoded iris templates found in iris_codes.csv. Run stage 6 first.")
    for row in rows:
        code_path = Path(row["code_path"])
        mask_path = Path(row["mask_path"])
        if not mask_path.exists():
            raise IrisPipelineError(f"Missing mask for template: {code_path.name}")
        code = np.load(code_path)
        mask = np.load(mask_path)
        stem = code_path.name.removesuffix("_iris_code.npy")
        if code.shape != mask.shape:
            raise IrisPipelineError(f"Code/mask shape mismatch for template: {stem}")
        if code.ndim != 3 or code.shape[2] != 2:
            raise IrisPipelineError(f"Unexpected code shape for template {stem}: {code.shape}")
        subject_id = row.get("subject_id", "").strip() or _subject_id_from_name(stem)
        image_name = row.get("image", "").strip() or f"{stem}.jpg"
        templates.append(IrisTemplate(image_name, subject_id, code.astype(np.uint8), mask.astype(np.uint8)))
    return templates


def _masked_hamming_for_shift(a: IrisTemplate, b: IrisTemplate, shift: int) -> tuple[float, int]:
    code_b = np.roll(b.code, shift=shift, axis=1)
    mask_b = np.roll(b.mask, shift=shift, axis=1)
    valid = (a.mask > 0) & (mask_b > 0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 1.0, 0
    diff = (a.code != code_b) & valid
    return float(np.count_nonzero(diff)) / float(valid_count), valid_count


def best_hamming_distance(a: IrisTemplate, b: IrisTemplate, max_shift: int) -> tuple[float, int, int]:
    best_distance, best_shift, best_valid_bits = 1.0, 0, 0
    for shift in range(-max_shift, max_shift + 1):
        distance, valid_bits = _masked_hamming_for_shift(a, b, shift)
        if distance < best_distance:
            best_distance, best_shift, best_valid_bits = distance, shift, valid_bits
    return best_distance, best_shift, best_valid_bits


def verify_pair(a: IrisTemplate, b: IrisTemplate, params: Stage7Params) -> dict[str, str | float | int | bool]:
    distance, shift, valid_bits = best_hamming_distance(a, b, params.max_angular_shift)
    return {
        "image_a": a.image_name,
        "image_b": b.image_name,
        "subject_a": a.subject_id,
        "subject_b": b.subject_id,
        "hamming_distance": distance,
        "best_shift": shift,
        "valid_bits": valid_bits,
        "is_genuine": a.subject_id == b.subject_id,
        "accepted": distance <= params.verify_threshold,
    }


def identify_probe(probe: IrisTemplate, gallery: list[IrisTemplate], params: Stage7Params) -> dict[str, str | float | int | bool]:
    if not gallery:
        raise IrisPipelineError("Gallery is empty in identify_probe.")
    best_match, best_distance, best_shift = gallery[0], 1.0, 0
    for candidate in gallery:
        distance, shift, _ = best_hamming_distance(probe, candidate, params.max_angular_shift)
        if distance < best_distance:
            best_distance, best_shift, best_match = distance, shift, candidate
    return {
        "probe_image": probe.image_name,
        "probe_subject": probe.subject_id,
        "match_image": best_match.image_name,
        "match_subject": best_match.subject_id,
        "hamming_distance": best_distance,
        "best_shift": best_shift,
        "is_correct": probe.subject_id == best_match.subject_id,
    }


def _write_csv(path: Path, rows: list[dict[str, str | float | int | bool]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_stage7_matching(app_config: AppConfig) -> None:
    params = _extract_stage7_params(app_config)
    templates = _load_templates(app_config.output_dir)
    if len(templates) < 2:
        raise IrisPipelineError("At least two templates are required for matching.")
    LOGGER.info("Stage 7 started for %d templates.", len(templates))
    verification_rows: list[dict[str, str | float | int | bool]] = []
    for idx_a in range(len(templates)):
        for idx_b in range(idx_a + 1, len(templates)):
            verification_rows.append(verify_pair(templates[idx_a], templates[idx_b], params))
    identification_rows: list[dict[str, str | float | int | bool]] = []
    for idx, probe in enumerate(templates):
        gallery = [t for j, t in enumerate(templates) if j != idx]
        identification_rows.append(identify_probe(probe, gallery, params))
    verify_path = app_config.output_dir / "verification_pairs.csv"
    identify_path = app_config.output_dir / "identification_results.csv"
    _write_csv(verify_path, verification_rows, ["image_a", "image_b", "subject_a", "subject_b", "hamming_distance", "best_shift", "valid_bits", "is_genuine", "accepted"])
    _write_csv(identify_path, identification_rows, ["probe_image", "probe_subject", "match_image", "match_subject", "hamming_distance", "best_shift", "is_correct"])
    id_accuracy = float(np.mean([1.0 if bool(row["is_correct"]) else 0.0 for row in identification_rows]))
    LOGGER.info("Stage 7 finished successfully. Identification accuracy (LOO): %.3f", id_accuracy)
    LOGGER.info("Saved: %s", verify_path)
    LOGGER.info("Saved: %s", identify_path)


def _extract_stage8_params(app_config: AppConfig) -> Stage8Params:
    evaluation = app_config.raw.get("evaluation", {})
    threshold_steps = evaluation.get("threshold_steps", 1001)
    if not isinstance(threshold_steps, int) or threshold_steps < 11:
        raise ConfigError("'evaluation.threshold_steps' must be integer >= 11.")
    return Stage8Params(threshold_steps)


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes"}


def _load_verification_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise IrisPipelineError(f"Missing verification file: {path}")
    genuine: list[float] = []
    impostor: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            distance = float(row["hamming_distance"])
            if _parse_bool(row["is_genuine"]):
                genuine.append(distance)
            else:
                impostor.append(distance)
    return np.asarray(genuine, dtype=np.float32), np.asarray(impostor, dtype=np.float32)


def _load_identification_accuracy(path: Path) -> float:
    if not path.exists():
        raise IrisPipelineError(f"Missing identification file: {path}")
    vals: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            vals.append(1.0 if _parse_bool(row["is_correct"]) else 0.0)
    if not vals:
        raise IrisPipelineError("Identification results are empty.")
    return float(np.mean(vals))


def _compute_far_frr(genuine: np.ndarray, impostor: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    thresholds = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    far = np.array([float(np.mean(impostor <= t)) for t in thresholds], dtype=np.float32)
    frr = np.array([float(np.mean(genuine > t)) for t in thresholds], dtype=np.float32)
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return thresholds, far, frr, eer, float(thresholds[idx])


def _draw_histogram(genuine: np.ndarray, impostor: np.ndarray, output_path: Path) -> None:
    width, height = 1000, 600
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin, bins = 60, 40
    g_hist, _ = np.histogram(genuine, bins=bins, range=(0.0, 1.0))
    i_hist, _ = np.histogram(impostor, bins=bins, range=(0.0, 1.0))
    max_count = max(int(g_hist.max()) if genuine.size else 0, int(i_hist.max()) if impostor.size else 0, 1)
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    bin_w = plot_w / bins
    for idx in range(bins):
        x0 = int(margin + idx * bin_w)
        x1 = int(margin + (idx + 1) * bin_w) - 1
        gh = int((g_hist[idx] / max_count) * plot_h)
        ih = int((i_hist[idx] / max_count) * plot_h)
        cv2.rectangle(canvas, (x0, height - margin - gh), (x1, height - margin), (70, 140, 240), -1)
        cv2.rectangle(canvas, (x0, height - margin - ih), (x1, height - margin), (60, 180, 75), 2)
    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (0, 0, 0), 1)
    cv2.putText(canvas, "Genuine (blue) vs Impostor (green) distances", (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(output_path), canvas)


def _draw_far_frr_curve(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray, eer_threshold: float, output_path: Path) -> None:
    width, height = 1000, 600
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin = 60
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    def to_xy(x_val: float, y_val: float) -> tuple[int, int]:
        return int(margin + x_val * plot_w), int(height - margin - y_val * plot_h)
    for idx in range(1, len(thresholds)):
        cv2.line(canvas, to_xy(float(thresholds[idx - 1]), float(far[idx - 1])), to_xy(float(thresholds[idx]), float(far[idx])), (50, 50, 220), 2)
        cv2.line(canvas, to_xy(float(thresholds[idx - 1]), float(frr[idx - 1])), to_xy(float(thresholds[idx]), float(frr[idx])), (220, 60, 60), 2)
    cv2.line(canvas, to_xy(eer_threshold, 0.0), to_xy(eer_threshold, 1.0), (0, 0, 0), 1)
    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (0, 0, 0), 1)
    cv2.putText(canvas, "FAR (blue) and FRR (red)", (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(output_path), canvas)


def run_stage8_evaluation(app_config: AppConfig) -> None:
    params = _extract_stage8_params(app_config)
    results_dir = app_config.output_dir
    genuine, impostor = _load_verification_pairs(results_dir / "verification_pairs.csv")
    id_accuracy = _load_identification_accuracy(results_dir / "identification_results.csv")
    can_compute = genuine.size > 0 and impostor.size > 0
    if can_compute:
        thresholds, far, frr, eer, eer_threshold = _compute_far_frr(genuine, impostor, params.threshold_steps)
    else:
        thresholds, far, frr, eer, eer_threshold = np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), None, None
    summary = {
        "genuine_pairs": int(genuine.size),
        "impostor_pairs": int(impostor.size),
        "genuine_mean_distance": float(np.mean(genuine)) if genuine.size else None,
        "impostor_mean_distance": float(np.mean(impostor)) if impostor.size else None,
        "genuine_std_distance": float(np.std(genuine)) if genuine.size else None,
        "impostor_std_distance": float(np.std(impostor)) if impostor.size else None,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "identification_accuracy_loo": id_accuracy,
        "notes": "Full FAR/FRR/EER computed." if can_compute else "Insufficient class diversity for FAR/FRR/EER (need both genuine and impostor pairs).",
    }
    summary_path = results_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_csv = results_dir / "far_frr_curve.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["threshold", "far", "frr"])
        writer.writeheader()
        if can_compute:
            for idx in range(len(thresholds)):
                writer.writerow({"threshold": float(thresholds[idx]), "far": float(far[idx]), "frr": float(frr[idx])})
    _draw_histogram(genuine, impostor, results_dir / "distance_histogram.png")
    if can_compute and eer_threshold is not None:
        _draw_far_frr_curve(thresholds, far, frr, eer_threshold, results_dir / "far_frr_curve.png")
    else:
        blank = np.full((350, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(blank, "FAR/FRR curve unavailable: need genuine and impostor pairs.", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.imwrite(str(results_dir / "far_frr_curve.png"), blank)
    if eer is not None and eer_threshold is not None:
        LOGGER.info("Stage 8 finished successfully. EER=%.4f at threshold=%.4f", eer, eer_threshold)
    else:
        LOGGER.info("Stage 8 finished successfully with partial metrics (single-class data).")
    LOGGER.info("Saved: %s", summary_path)
    LOGGER.info("Saved: %s", metrics_csv)

