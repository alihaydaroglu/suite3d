import numpy as np

from suite3d.io.lbmio import (
    arrange_lbm_planes_for_analysis,
    infer_lbm_cavity_size_from_planes,
    lbm_plane_metadata,
)


def test_arranges_single_even_cavity_lowest_to_highest():
    assert arrange_lbm_planes_for_analysis([8, 0, 4, 2]) == [0, 2, 4, 8]


def test_arranges_single_odd_cavity_lowest_to_highest():
    assert arrange_lbm_planes_for_analysis([7, 1, 5, 3]) == [1, 3, 5, 7]


def test_arranges_two_cavities_even_then_odd():
    assert arrange_lbm_planes_for_analysis([5, 2, 1, 4, 0, 3]) == [
        0,
        2,
        4,
        1,
        3,
        5,
    ]


def test_keeps_consecutive_one_cavity_lowest_to_highest():
    assert arrange_lbm_planes_for_analysis(
        [3, 0, 2, 1],
        two_cavity=False,
    ) == [0, 1, 2, 3]


def test_infers_unequal_demo_cavity_size_from_first_cavity():
    planes = np.array(
        [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
            17,
        ]
    )
    assert infer_lbm_cavity_size_from_planes(planes) == 13


def test_single_cavity_disables_crosstalk_by_matching_plane_count():
    planes = [1, 3, 5, 7]
    assert infer_lbm_cavity_size_from_planes(planes) == len(planes)


def test_consecutive_one_cavity_disables_crosstalk_when_marked_one_cavity():
    planes = [0, 1, 2, 3]
    assert infer_lbm_cavity_size_from_planes(planes, two_cavity=False) == len(planes)


def test_plane_metadata_reports_raw_planes_and_cavities_in_z_order():
    raw_planes, cavity_labels = lbm_plane_metadata([3, 0, 1, 2])
    assert raw_planes == [0, 2, 1, 3]
    assert cavity_labels == ["A", "A", "B", "B"]
