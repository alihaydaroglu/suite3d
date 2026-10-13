import json

import numpy as np
import tifffile
from scipy.ndimage import shift as ndi_shift

from suite3d.viewer.raw_tiff import (
    RawTiffMovie,
    apply_xy_shifts_to_planes,
    estimate_plane_xy_shifts,
    mean_images_by_plane,
    percentile_images_by_plane,
)


def scanimage_artist_rois():
    rois = []
    for center_x in (-1.0, 1.0, 0.0):
        rois.append(
            {
                "zs": 0,
                "scanfields": {
                    "roiUuid": f"roi-{center_x}",
                    "centerXY": [center_x, 0.0],
                    "sizeXY": [1.0, 1.0],
                    "pixelResolutionXY": [2, 2],
                },
            }
        )
    return json.dumps({"RoiGroups": {"imagingRoiGroup": {"rois": rois}}})


def write_raw_tiff(path, offset=0, nt=2):
    data = np.zeros((1, nt, 3, 6, 2), dtype=np.int16)
    for t in range(data.shape[1]):
        for c in range(data.shape[2]):
            data[0, t, c, 0:2, :] = offset + 10 + 100 * t + c
            data[0, t, c, 2:4, :] = offset + 20 + 100 * t + c
            data[0, t, c, 4:6, :] = offset + 30 + 100 * t + c

    tifffile.imwrite(
        path,
        data,
        metadata={"axes": "ZTCYX"},
        software=(
            "SI.hChannels.channelSave = [5 1 3]\n"
            "SI.objectiveResolution = 160\n"
            "SI.hStackManager.stackZStepSize = 3\n"
        ),
        extratags=[(315, "s", 0, scanimage_artist_rois(), True)],
    )


def test_raw_tiff_movie_stitches_scanimage_strips_into_suite3d_fov(tmp_path):
    path = tmp_path / "raw.tif"
    write_raw_tiff(path)

    movie = RawTiffMovie(path)
    try:
        assert movie.shape == (3, 2, 2, 6)
        assert movie.plane_label(0) == "Channel 0 (raw 0, cavity A)"
        assert movie.plane_label(2) == "Channel 2 (raw 4, cavity A)"
        assert movie.xy_size_um == (480.0, 160.0)
        assert movie.voxel_size_um == (3.0, 80.0, 80.0)

        frame = movie.frame(global_frame=1, plane=0)
        expected = np.array(
            [
                [111, 111, 131, 131, 121, 121],
                [111, 111, 131, 131, 121, 121],
            ],
            dtype=frame.dtype,
        )
        np.testing.assert_array_equal(frame, expected)
    finally:
        movie.close()


def test_raw_tiff_movie_concatenates_folder_tiffs_over_time(tmp_path):
    write_raw_tiff(tmp_path / "raw_00002.tif", offset=1000, nt=2)
    write_raw_tiff(tmp_path / "raw_00001.tif", offset=0, nt=2)

    movie = RawTiffMovie(tmp_path)
    try:
        assert [path.name for path in movie.paths] == ["raw_00001.tif", "raw_00002.tif"]
        assert movie.shape == (3, 4, 2, 6)

        frame = movie.frame(global_frame=2, plane=0)
        expected = np.array(
            [
                [1011, 1011, 1031, 1031, 1021, 1021],
                [1011, 1011, 1031, 1031, 1021, 1021],
            ],
            dtype=frame.dtype,
        )
        np.testing.assert_array_equal(frame, expected)
    finally:
        movie.close()


def test_mean_images_by_plane_averages_frames(tmp_path):
    write_raw_tiff(tmp_path / "raw.tif", nt=2)

    movie = RawTiffMovie(tmp_path / "raw.tif")
    try:
        means = mean_images_by_plane(movie)
        assert means.shape == (3, 2, 6)
        expected_plane0 = np.array(
            [
                [61, 61, 81, 81, 71, 71],
                [61, 61, 81, 81, 71, 71],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(means[0], expected_plane0)
    finally:
        movie.close()


def test_percentile_images_by_plane_projects_temporal_percentile(tmp_path):
    write_raw_tiff(tmp_path / "raw.tif", nt=2)

    movie = RawTiffMovie(tmp_path / "raw.tif")
    try:
        p99_images = percentile_images_by_plane(movie, percentile=99.0)
        assert p99_images.shape == (3, 2, 6)
        expected_plane0 = np.array(
            [
                [110.5, 110.5, 130.5, 130.5, 120.5, 120.5],
                [110.5, 110.5, 130.5, 130.5, 120.5, 120.5],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(p99_images[0], expected_plane0)
    finally:
        movie.close()


def test_estimate_plane_xy_shifts_returns_shift_to_align_to_reference():
    y, x = np.indices((48, 48), dtype=np.float32)
    reference = (
        np.exp(-((y - 14) ** 2 + (x - 19) ** 2) / 18.0)
        + 0.7 * np.exp(-((y - 31) ** 2 + (x - 34) ** 2) / 10.0)
    ).astype(np.float32)
    moving = ndi_shift(reference, shift=(3, -2), order=1, mode="constant", cval=0.0)
    mean_images = np.stack([moving, reference])

    shifts, errors, reference_plane = estimate_plane_xy_shifts(
        mean_images,
        reference_plane=1,
        upsample_factor=10,
        max_shift_px=10,
    )

    assert reference_plane == 1
    np.testing.assert_allclose(shifts[1], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(shifts[0], [-3.0, 2.0], atol=0.25)
    assert np.isfinite(errors).all()


def test_apply_xy_shifts_to_planes_aligns_shifted_image():
    image = np.zeros((24, 24), dtype=np.float32)
    image[8:12, 11:15] = 1.0
    moving = ndi_shift(image, shift=(3, -2), order=0, mode="constant", cval=0.0)
    registered = apply_xy_shifts_to_planes(
        np.stack([moving, image]),
        np.array([[-3.0, 2.0], [0.0, 0.0]], dtype=np.float32),
        order=0,
    )

    np.testing.assert_allclose(registered[0], image, atol=1e-6)
    np.testing.assert_allclose(registered[1], image, atol=1e-6)
