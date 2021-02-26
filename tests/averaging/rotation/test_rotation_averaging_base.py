"""Tests for rotation averaging base class.

Authors: Ayush Baid
"""

import pickle
import unittest

import dask
import numpy as np
from gtsam import Rot3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.rotation.dummy_rotation_averaging import DummyRotationAveraging


class TestRotationAveragingBase(unittest.TestCase):
    """Main tests for rotation averaging base class."""

    def setUp(self):
        super(TestRotationAveragingBase, self).setUp()

        self.obj = DummyRotationAveraging()

    def test_circle_two_edges(self):
        """Tests for 4 poses in a circle, with a pose connected to its immediate neighborhood."""

        if isinstance(self.obj, DummyRotationAveraging):
            self.skipTest("Test case invalid for dummy class")

        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_TWO_EDGES_RELATIVE_POSES
        )
        wRi_computed = self.obj.run(len(wRi_expected), i2Ri1_dict)
        self.assertTrue(geometry_comparisons.compare_rotations(wRi_computed, wRi_expected))

    def test_circle_all_edges(self):
        """Tests for 4 poses in a circle, with a pose connected all others."""

        if isinstance(self.obj, DummyRotationAveraging):
            self.skipTest("Test case invalid for dummy class")

        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_ALL_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_ALL_EDGES_RELATIVE_POSES
        )
        wRi_computed = self.obj.run(len(wRi_expected), i2Ri1_dict)
        self.assertTrue(geometry_comparisons.compare_rotations(wRi_computed, wRi_expected))

    def test_line_large_edges(self):
        """Tests for 3 poses in a line, with large translations between them."""

        if isinstance(self.obj, DummyRotationAveraging):
            self.skipTest("Test case invalid for dummy class")

        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.LINE_LARGE_EDGES_GLOBAL_POSES, sample_poses.LINE_LARGE_EDGES_RELATIVE_POSES
        )
        wRi_computed = self.obj.run(len(wRi_expected), i2Ri1_dict)
        self.assertTrue(geometry_comparisons.compare_rotations(wRi_computed, wRi_expected))

    def test_panorama(self):
        """Tests for 3 poses in a panorama configuration (large rotations at the same location)"""

        if isinstance(self.obj, DummyRotationAveraging):
            self.skipTest("Test case invalid for dummy class")

        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.PANORAMA_GLOBAL_POSES, sample_poses.PANORAMA_RELATIVE_POSES
        )
        wRi_computed = self.obj.run(len(wRi_expected), i2Ri1_dict)
        self.assertTrue(geometry_comparisons.compare_rotations(wRi_computed, wRi_expected))

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection of relative poses."""

        num_poses = 3

        i2Ri1_dict = {
            (0, 1): Rot3.RzRyRx(0, np.deg2rad(30), 0),
            (1, 2): Rot3.RzRyRx(0, 0, np.deg2rad(20)),
        }

        i2Ri1_graph = dask.delayed(i2Ri1_dict)

        # use the GTSAM API directly (without dask) for rotation averaging
        expected_wRi_list = self.obj.run(num_poses, i2Ri1_dict)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(num_poses, i2Ri1_graph)

        with dask.config.set(scheduler="single-threaded"):
            wRi_list = dask.compute(computation_graph)[0]

        # compare the two results
        self.assertTrue(geometry_comparisons.compare_rotations(wRi_list, expected_wRi_list))

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == "__main__":
    unittest.main()
