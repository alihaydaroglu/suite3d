import tifffile

from suite3d.io import get_tif_paths


def write_tiff(path):
    tifffile.imwrite(path, [[1]])


def test_get_tif_paths_accepts_multiple_directories_and_tiff_files(tmp_path):
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    write_tiff(folder_a / "file_2.tif")
    write_tiff(folder_b / "file_1.tiff")
    write_tiff(tmp_path / "explicit.TIF")

    paths = get_tif_paths(
        [folder_a, folder_b, tmp_path / "explicit.TIF"],
        natsort=True,
    )

    assert sorted(path.split("\\")[-1].split("/")[-1] for path in paths) == [
        "explicit.TIF",
        "file_1.tiff",
        "file_2.tif",
    ]


def test_get_tif_paths_can_look_one_level_down(tmp_path):
    root = tmp_path / "root"
    child = root / "child"
    grandchild = child / "grandchild"
    grandchild.mkdir(parents=True)
    write_tiff(root / "root.tif")
    write_tiff(child / "child.tif")
    write_tiff(grandchild / "grandchild.tif")

    paths = get_tif_paths(root, look_one_level_down=True, natsort=True)

    assert [path.split("\\")[-1].split("/")[-1] for path in paths] == [
        "child.tif",
        "root.tif",
    ]


def test_get_tif_paths_deduplicates_overlapping_inputs(tmp_path):
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    write_tiff(child / "child.tif")

    paths = get_tif_paths([root, child], look_one_level_down=True)

    assert len(paths) == 1
    assert paths[0].endswith("child.tif")
