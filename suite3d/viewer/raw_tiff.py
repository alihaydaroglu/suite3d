from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import numpy as np
import tifffile

from suite3d.io.lbmio import (
    _split_rois_from_tif,
    _stitch_rois_fast,
    arrange_lbm_planes_for_analysis,
    get_meso_rois,
)


@dataclass(frozen=True)
class RawTiffPlane:
    indices: dict[str, int]
    label: str


@dataclass(frozen=True)
class RawTiffFile:
    path: Path
    tif: tifffile.TiffFile
    axes: str
    series_shape: tuple[int, ...]
    leading_axes: str
    leading_shape: tuple[int, ...]
    axis_sizes: dict[str, int]
    frame_count: int


def _scanimage_channel_save(software: str) -> list[int] | None:
    match = re.search(r"SI\.hChannels\.channelSave\s*=\s*\[([^\]]+)\]", software)
    if match is None:
        return None
    values = [int(value) for value in re.findall(r"-?\d+", match.group(1))]
    if not values:
        return None
    return values


def _scanimage_number(software: str, key: str) -> float | None:
    match = re.search(re.escape(key) + r"\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", software)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _series_axes_and_shape(tif: tifffile.TiffFile) -> tuple[str, tuple[int, ...]]:
    series = tif.series[0]
    axes = str(series.axes)
    shape = tuple(int(size) for size in series.shape)
    if len(shape) < 2 or axes[-2:] != "YX":
        page = tif.pages[0]
        return "QYX", (len(tif.pages), int(page.shape[0]), int(page.shape[1]))
    return axes, shape


def _natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def raw_tiff_paths(path: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(path, (str, Path)):
        path = Path(path)
        if path.is_dir():
            paths = [
                candidate
                for candidate in path.iterdir()
                if candidate.is_file() and candidate.suffix.lower() in (".tif", ".tiff")
            ]
            paths = sorted(paths, key=_natural_key)
            if not paths:
                raise FileNotFoundError(f"No .tif/.tiff files found in {path}")
            return paths
        return [path]

    paths = [Path(candidate) for candidate in path]
    if not paths:
        raise FileNotFoundError("No raw TIFF files were provided")
    return sorted(paths, key=_natural_key)


def _frame_count(axes: str, shape: tuple[int, ...]) -> int:
    leading_axes = axes[:-2]
    leading_shape = shape[:-2]
    frame_shape = tuple(
        size for axis, size in zip(leading_axes, leading_shape) if axis != "C"
    )
    if not frame_shape:
        return max(1, int(dict(zip(leading_axes, leading_shape)).get("Q", 1)))
    return int(np.prod(frame_shape))


def _actual_frame_count(tif: tifffile.TiffFile, axes: str, shape: tuple[int, ...]) -> int:
    frame_count = _frame_count(axes, shape)
    leading_axes = axes[:-2]
    leading_shape = shape[:-2]
    expected_pages = int(np.prod(leading_shape)) if leading_shape else 1
    page_count = len(tif.pages)
    if page_count == expected_pages:
        return frame_count

    axis_sizes = dict(zip(leading_axes, leading_shape))
    if "C" in axis_sizes:
        channel_count = int(axis_sizes["C"])
        if channel_count > 0 and page_count % channel_count == 0:
            return max(1, int(page_count // channel_count))

    if "T" in axis_sizes and page_count == int(axis_sizes["T"]):
        return int(axis_sizes["T"])

    return max(1, min(frame_count, page_count))


class RawTiffMovie:
    """On-demand raw TIFF movie in Suite3D viewer coordinates.

    ScanImage LBM files store multiple strip FOVs vertically in each TIFF page.
    Suite3D analysis splits those strips and places them back into physical FOV
    coordinates before registration.  This viewer source does the same per page,
    instead of showing the raw concatenated strip image.
    """

    def __init__(self, path: Path | str | Sequence[Path | str], cache_size: int = 64) -> None:
        self.paths = raw_tiff_paths(path)
        self.path = self.paths[0].parent if len(self.paths) > 1 else self.paths[0]
        self.files: list[RawTiffFile] = []
        try:
            for file_path in self.paths:
                tif = tifffile.TiffFile(file_path)
                axes, series_shape = _series_axes_and_shape(tif)
                leading_axes = axes[:-2]
                leading_shape = series_shape[:-2]
                self.files.append(
                    RawTiffFile(
                        path=file_path,
                        tif=tif,
                        axes=axes,
                        series_shape=series_shape,
                        leading_axes=leading_axes,
                        leading_shape=leading_shape,
                        axis_sizes=dict(zip(leading_axes, leading_shape)),
                        frame_count=_actual_frame_count(tif, axes, series_shape),
                    )
                )
        except Exception:
            self.close()
            raise

        first = self.files[0]
        self.tif = first.tif
        self.axes = first.axes
        self.series_shape = first.series_shape
        self.leading_axes = first.leading_axes
        self.leading_shape = first.leading_shape
        self.axis_sizes = first.axis_sizes
        self._validate_sequence()
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

        self.software = ""
        page0 = first.tif.pages[0]
        if "Software" in page0.tags:
            self.software = str(page0.tags["Software"].value)

        self.scanimage_channels = _scanimage_channel_save(self.software)
        self.rois = self._load_rois()
        self.planes = self._build_planes()
        self.frame_counts = [file.frame_count for file in self.files]
        self.frame_starts = np.concatenate([[0], np.cumsum(self.frame_counts)]).astype(int)
        self.frame_count = int(self.frame_starts[-1])
        sample = self._stitch_page(self._array_for_indices(first, self.planes[0].indices, 0))
        self.shape = (
            len(self.planes),
            self.frame_count,
            int(sample.shape[0]),
            int(sample.shape[1]),
        )
        self.z_step_um = _scanimage_number(self.software, "SI.hStackManager.stackZStepSize")
        self.xy_size_um = self._infer_xy_size_um()
        self.voxel_size_um = self._infer_voxel_size_um()

    def close(self) -> None:
        if hasattr(self, "_cache"):
            self._cache.clear()
        for file in getattr(self, "files", []):
            file.tif.close()
        self.files = []

    def _validate_sequence(self) -> None:
        first = self.files[0]
        comparable_axes = first.axes.replace("T", "").replace("Q", "").replace("Z", "")
        comparable_shape = tuple(
            size
            for axis, size in zip(first.axes, first.series_shape)
            if axis not in ("T", "Q", "Z")
        )
        for file in self.files[1:]:
            axes = file.axes.replace("T", "").replace("Q", "").replace("Z", "")
            shape = tuple(
                size
                for axis, size in zip(file.axes, file.series_shape)
                if axis not in ("T", "Q", "Z")
            )
            if axes != comparable_axes or shape != comparable_shape:
                raise ValueError(
                    "Raw TIFF sequence files must have matching channel and image dimensions; "
                    f"{file.path.name} is {file.axes} {file.series_shape}, "
                    f"expected compatible with {first.axes} {first.series_shape}"
                )

    def _load_rois(self) -> list[dict]:
        try:
            return get_meso_rois(str(self.paths[0]))
        except Exception:
            return []

    def _infer_xy_size_um(self) -> tuple[float, float] | None:
        objective_resolution = _scanimage_number(self.software, "SI.objectiveResolution")
        if objective_resolution is None or objective_resolution <= 0 or not self.rois:
            return None

        try:
            centers = np.asarray([roi["center"] for roi in self.rois], dtype=float)
            sizes = np.asarray([roi["sizeXY"] for roi in self.rois], dtype=float)
        except Exception:
            return None
        if centers.ndim != 2 or sizes.ndim != 2 or centers.shape[1] < 2 or sizes.shape[1] < 2:
            return None

        mins = np.nanmin(centers[:, :2] - sizes[:, :2] / 2.0, axis=0)
        maxs = np.nanmax(centers[:, :2] + sizes[:, :2] / 2.0, axis=0)
        xy_um = (maxs - mins) * float(objective_resolution)
        if not np.all(np.isfinite(xy_um)) or np.any(xy_um <= 0):
            return None
        return float(xy_um[0]), float(xy_um[1])

    def _infer_voxel_size_um(self) -> tuple[float, float, float] | None:
        if self.xy_size_um is None:
            return None
        nx = int(self.shape[3])
        ny = int(self.shape[2])
        if nx <= 0 or ny <= 0:
            return None
        x_um, y_um = self.xy_size_um
        z_um = float(self.z_step_um) if self.z_step_um is not None and self.z_step_um > 0 else 1.0
        return z_um, y_um / ny, x_um / nx

    def dimension_summary_um(self) -> str | None:
        parts = []
        if self.xy_size_um is not None:
            x_um, y_um = self.xy_size_um
            parts.append(f"x {x_um:g} um")
            parts.append(f"y {y_um:g} um")
        if self.z_step_um is not None and self.z_step_um > 0:
            z_span = max(0, self.frame_count - 1) * float(self.z_step_um)
            parts.append(f"z step {self.z_step_um:g} um")
            parts.append(f"z span {z_span:g} um")
        return ", ".join(parts) if parts else None

    def _build_planes(self) -> list[RawTiffPlane]:
        c_count = int(self.axis_sizes.get("C", 1))

        channel_ids = self.scanimage_channels
        if channel_ids is not None and len(channel_ids) == c_count:
            raw_plane_ids = [channel - 1 for channel in channel_ids]
            ordered_raw_plane_ids = arrange_lbm_planes_for_analysis(raw_plane_ids)
            c_order = [raw_plane_ids.index(raw) for raw in ordered_raw_plane_ids]
        else:
            raw_plane_ids = list(range(c_count))
            c_order = list(range(c_count))
        planes: list[RawTiffPlane] = []
        for c in c_order:
            indices: dict[str, int] = {}
            if "C" in self.axis_sizes:
                indices["C"] = c
            if "C" in self.axis_sizes:
                raw = raw_plane_ids[c]
                cavity = "A" if raw % 2 == 0 else "B"
                label = f"raw {raw}, cavity {cavity}"
            else:
                label = "raw page"
            planes.append(RawTiffPlane(indices=indices, label=label))

        if not planes:
            planes.append(RawTiffPlane(indices={}, label="raw page"))
        return planes

    def _file_and_local_frame(self, global_frame: int) -> tuple[RawTiffFile, int]:
        frame = max(0, min(int(global_frame), self.frame_count - 1))
        file_idx = int(np.searchsorted(self.frame_starts, frame, side="right") - 1)
        local_frame = int(frame - self.frame_starts[file_idx])
        return self.files[file_idx], local_frame

    def _frame_indices(self, file: RawTiffFile, frame: int) -> dict[str, int]:
        frame_axes = [
            axis for axis in file.leading_axes if axis != "C"
        ]
        frame_shape = tuple(file.axis_sizes[axis] for axis in frame_axes)
        if not frame_axes:
            return {}
        frame = max(0, min(int(frame), int(np.prod(frame_shape)) - 1))
        return dict(zip(frame_axes, np.unravel_index(frame, frame_shape)))

    def _page_index(self, file: RawTiffFile, indices: dict[str, int], frame: int) -> int:
        if file.leading_axes == "Q":
            return max(0, min(int(frame), len(file.tif.pages) - 1))

        frame_indices = self._frame_indices(file, frame)
        page_indices = []
        for axis, size in zip(file.leading_axes, file.leading_shape):
            if axis == "C":
                value = int(indices.get(axis, 0))
            else:
                value = int(frame_indices.get(axis, 0))
            page_indices.append(max(0, min(value, int(size) - 1)))

        if not page_indices:
            return 0
        return int(np.ravel_multi_index(tuple(page_indices), file.leading_shape))

    def _page_for_indices(self, file: RawTiffFile, indices: dict[str, int], frame: int):
        return file.tif.pages[self._page_index(file, indices, frame)]

    def _array_for_indices(self, file: RawTiffFile, indices: dict[str, int], frame: int) -> np.ndarray:
        expected_pages = int(np.prod(file.leading_shape)) if file.leading_shape else 1
        if len(file.tif.pages) == expected_pages:
            return self._page_for_indices(file, indices, frame).asarray()

        if "T" in file.axis_sizes and len(file.tif.pages) == file.axis_sizes["T"]:
            page = np.asarray(file.tif.pages[max(0, min(int(frame), len(file.tif.pages) - 1))].asarray())
            page_axes = [axis for axis in file.leading_axes if axis != "T"]
            while len(page_axes) > max(0, page.ndim - 2):
                removable = next((idx for idx, axis in enumerate(page_axes) if file.axis_sizes.get(axis, 1) == 1), None)
                if removable is None:
                    break
                page_axes.pop(removable)
            if len(page_axes) == page.ndim - 2:
                selector = tuple(indices.get(axis, 0) for axis in page_axes)
                return page[selector]

        return self._page_for_indices(file, indices, frame).asarray()

    def _stitch_page(self, page: np.ndarray) -> np.ndarray:
        page = np.asarray(page)
        if page.ndim != 2 or len(self.rois) <= 1:
            return page
        split = _split_rois_from_tif(page[np.newaxis, np.newaxis, :, :], self.rois, ch_id=0)
        return np.asarray(_stitch_rois_fast(split, self.rois)[0])

    def frame(self, global_frame: int, plane: int) -> np.ndarray:
        plane = max(0, min(int(plane), len(self.planes) - 1))
        frame = max(0, min(int(global_frame), self.frame_count - 1))
        key = (plane, frame)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        file, local_frame = self._file_and_local_frame(frame)
        page = self._array_for_indices(file, self.planes[plane].indices, local_frame)
        image = self._stitch_page(page)
        self._cache[key] = image
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image

    def y_slice(self, global_frame: int, y_index: int) -> np.ndarray:
        y_index = int(y_index)
        planes = [self.frame(global_frame, plane) for plane in range(len(self.planes))]
        y_index = max(0, min(y_index, int(planes[0].shape[0]) - 1))
        return np.stack([plane[y_index, :] for plane in planes], axis=0)

    def plane_label(self, plane: int) -> str:
        plane = int(plane)
        if plane < 0 or plane >= len(self.planes):
            return f"Channel {plane}"
        return f"Channel {plane} ({self.planes[plane].label})"


def mean_images_by_plane(
    movie: RawTiffMovie,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Compute one mean image per plane across all raw TIFF frames."""
    nz, nt, ny, nx = movie.shape
    means = np.empty((int(nz), int(ny), int(nx)), dtype=np.float32)
    for plane in range(int(nz)):
        acc = np.zeros((int(ny), int(nx)), dtype=np.float64)
        for frame in range(int(nt)):
            acc += np.asarray(movie.frame(frame, plane), dtype=np.float64)
            if progress_callback is not None:
                progress_callback(plane, frame + 1)
        means[plane] = (acc / max(int(nt), 1)).astype(np.float32, copy=False)
    return means


def max_images_by_plane(
    movie: RawTiffMovie,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Compute one max-projection image per plane across all raw TIFF frames."""
    nz, nt, ny, nx = movie.shape
    max_images = np.empty((int(nz), int(ny), int(nx)), dtype=np.float32)
    for plane in range(int(nz)):
        projection = None
        for frame in range(int(nt)):
            image = np.asarray(movie.frame(frame, plane), dtype=np.float32)
            if projection is None:
                projection = image.copy()
            else:
                np.maximum(projection, image, out=projection)
            if progress_callback is not None:
                progress_callback(plane, frame + 1)
        if projection is None:
            projection = np.zeros((int(ny), int(nx)), dtype=np.float32)
        max_images[plane] = projection
    return max_images


def percentile_images_by_plane(
    movie: RawTiffMovie,
    percentile: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Compute one temporal percentile-projection image per plane."""
    nz, nt, ny, nx = movie.shape
    if not 0.0 <= float(percentile) <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    projections = np.empty((int(nz), int(ny), int(nx)), dtype=np.float32)
    for plane in range(int(nz)):
        frames = np.empty((int(nt), int(ny), int(nx)), dtype=np.float32)
        for frame in range(int(nt)):
            frames[frame] = np.asarray(movie.frame(frame, plane), dtype=np.float32)
            if progress_callback is not None:
                progress_callback(plane, frame + 1)
        projections[plane] = np.percentile(frames, float(percentile), axis=0).astype(
            np.float32,
            copy=False,
        )
    return projections


def _parabolic_peak_offset(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return 0.0
    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])
    denom = left - 2.0 * center + right
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return 0.0
    return float(offset)


def _bounded_phase_correlation_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_px: int,
) -> tuple[np.ndarray, float]:
    from scipy.fft import fft2, fftshift, ifft2

    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    cross_power = fft2(reference) * np.conj(fft2(moving))
    magnitude = np.abs(cross_power)
    cross_power = np.divide(
        cross_power,
        magnitude,
        out=np.zeros_like(cross_power),
        where=magnitude > 1e-12,
    )
    correlation = fftshift(ifft2(cross_power).real)
    cy = correlation.shape[0] // 2
    cx = correlation.shape[1] // 2
    max_shift = max(0, int(max_shift_px))
    y0 = max(0, cy - max_shift)
    y1 = min(correlation.shape[0], cy + max_shift + 1)
    x0 = max(0, cx - max_shift)
    x1 = min(correlation.shape[1], cx + max_shift + 1)
    search = correlation[y0:y1, x0:x1]
    if search.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32), float("nan")

    local_y, local_x = np.unravel_index(int(np.argmax(search)), search.shape)
    peak_y = int(y0 + local_y)
    peak_x = int(x0 + local_x)
    dy = float(peak_y - cy)
    dx = float(peak_x - cx)
    dy += _parabolic_peak_offset(correlation[:, peak_x], peak_y)
    dx += _parabolic_peak_offset(correlation[peak_y, :], peak_x)
    return np.array([dy, dx], dtype=np.float32), float(correlation[peak_y, peak_x])


def estimate_plane_xy_shifts(
    plane_images: np.ndarray,
    reference_plane: int | None = None,
    upsample_factor: int = 10,
    max_shift_px: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate y/x shifts aligning each plane image to a reference plane.

    Returns:
        shifts: ``(n_planes, 2)`` array in ``(y, x)`` pixels. Each row is the
            shift to apply to that plane image to align it to the reference.
        errors: Registration error reported by phase cross-correlation, or
            ``1 - bounded_phase_correlation_peak`` when ``max_shift_px`` is set.
        reference_plane: Reference plane index used for registration.
    """
    images = np.asarray(plane_images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("plane_images must have shape (planes, y, x)")
    if images.shape[0] == 0:
        raise ValueError("plane_images has no planes")

    if reference_plane is None:
        reference_plane = images.shape[0] // 2
    reference_plane = int(reference_plane)
    if not 0 <= reference_plane < images.shape[0]:
        raise ValueError(
            f"reference_plane={reference_plane} is outside "
            f"0..{images.shape[0] - 1}"
        )

    if max_shift_px is None:
        try:
            from skimage.registration import phase_cross_correlation
        except Exception as exc:
            raise RuntimeError(
                "Plane cross-correlation registration requires scikit-image."
            ) from exc

    reference = np.asarray(images[reference_plane], dtype=np.float32)
    reference = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
    reference = reference - float(np.mean(reference))

    shifts = np.zeros((images.shape[0], 2), dtype=np.float32)
    errors = np.full(images.shape[0], np.nan, dtype=np.float32)
    for plane in range(images.shape[0]):
        moving = np.asarray(images[plane], dtype=np.float32)
        moving = np.nan_to_num(moving, nan=0.0, posinf=0.0, neginf=0.0)
        moving = moving - float(np.mean(moving))
        if plane == reference_plane:
            errors[plane] = 0.0
            continue
        if max_shift_px is None:
            shift, error, _phase = phase_cross_correlation(
                reference,
                moving,
                upsample_factor=max(1, int(upsample_factor)),
            )
            shifts[plane] = np.asarray(shift[:2], dtype=np.float32)
            errors[plane] = float(error)
        else:
            shift, peak = _bounded_phase_correlation_shift(
                reference,
                moving,
                max_shift_px=max_shift_px,
            )
            shifts[plane] = shift
            errors[plane] = 1.0 - peak if np.isfinite(peak) else np.nan

    return shifts, errors, reference_plane


def apply_xy_shifts_to_planes(
    images: np.ndarray,
    shifts: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Apply per-plane ``(y, x)`` shifts to a stack of plane images."""
    images = np.asarray(images, dtype=np.float32)
    shifts = np.asarray(shifts, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("images must have shape (planes, y, x)")
    if shifts.shape != (images.shape[0], 2):
        raise ValueError(
            f"shifts must have shape ({images.shape[0]}, 2), got {shifts.shape}"
        )

    from scipy.ndimage import shift as ndi_shift

    shifted = np.empty_like(images, dtype=np.float32)
    for plane in range(images.shape[0]):
        shifted[plane] = ndi_shift(
            images[plane],
            shift=tuple(float(v) for v in shifts[plane]),
            order=int(order),
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32, copy=False)
    return shifted
